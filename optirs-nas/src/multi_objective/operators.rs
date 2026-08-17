//! Architecture-level variation operators shared by the population-based
//! optimizers in this module (F18).
//!
//! Before this existed, `NSGA2::generate_random_architecture` returned the *same*
//! single-component Adam architecture every call, and both `crossover` and
//! `mutate` touched only the `parameters` map — the `components` vector, which is
//! what actually determines the optimizer being searched for, was never varied.
//! A "population" was therefore one architecture repeated `population_size`
//! times, and the genetic operators could not move the search at all.
//!
//! Everything here draws from an explicitly passed RNG so a seeded optimizer
//! stays reproducible, and every architecture the sampler produces satisfies the
//! invariants the mutation/crossover operators assume (non-empty `components`,
//! `parameters` and `hyperparameters` agreeing, chain `connections`).

use crate::nas_engine::OptimizerArchitecture;
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use std::collections::HashMap;
use std::fmt::Debug;

/// Vocabulary of optimizer component labels used to seed and mutate candidate
/// architectures. Kept small but diverse so the genetic operators have
/// meaningful material to recombine; the strings match the `Debug` names of
/// [`crate::architecture::ComponentType`] variants, which is the convention the
/// rest of the crate (encoders, benchmark family resolution) reads.
pub const OPTIMIZER_COMPONENT_VOCAB: [&str; 7] = [
    "SGD", "Adam", "AdamW", "RMSprop", "AdaGrad", "Momentum", "Lion",
];

/// Minimum number of components a sampled architecture carries.
pub const MIN_SAMPLED_COMPONENTS: usize = 1;

/// Maximum number of components a sampled architecture carries.
pub const MAX_SAMPLED_COMPONENTS: usize = 4;

/// A searchable continuous hyperparameter: name, inclusive bounds, and whether
/// the range should be sampled on a logarithmic scale.
struct HyperparameterSpec {
    name: &'static str,
    min: f64,
    max: f64,
    log_scale: bool,
}

/// Continuous hyperparameters attached to every sampled architecture. The ranges
/// are the conventional search ranges for these quantities, so a sampled
/// architecture is a plausible optimizer configuration rather than a fixed
/// triple of constants.
const HYPERPARAMETER_SPECS: [HyperparameterSpec; 6] = [
    HyperparameterSpec {
        name: "learning_rate",
        min: 1e-5,
        max: 1e-1,
        log_scale: true,
    },
    HyperparameterSpec {
        name: "beta1",
        min: 0.5,
        max: 0.99,
        log_scale: false,
    },
    HyperparameterSpec {
        name: "beta2",
        min: 0.9,
        max: 0.9999,
        log_scale: false,
    },
    HyperparameterSpec {
        name: "epsilon",
        min: 1e-10,
        max: 1e-6,
        log_scale: true,
    },
    HyperparameterSpec {
        name: "weight_decay",
        min: 0.0,
        max: 0.1,
        log_scale: false,
    },
    HyperparameterSpec {
        name: "momentum",
        min: 0.0,
        max: 0.99,
        log_scale: false,
    },
];

/// Names of the hyperparameters every sampled architecture carries.
pub fn hyperparameter_names() -> Vec<&'static str> {
    HYPERPARAMETER_SPECS.iter().map(|spec| spec.name).collect()
}

/// Inclusive `[min, max]` bounds for a known hyperparameter name.
pub fn hyperparameter_bounds(name: &str) -> Option<(f64, f64)> {
    HYPERPARAMETER_SPECS
        .iter()
        .find(|spec| spec.name == name)
        .map(|spec| (spec.min, spec.max))
}

fn sample_hyperparameter<T: Float>(
    spec: &HyperparameterSpec,
    rng: &mut Random<scirs2_core::random::rngs::StdRng>,
) -> T {
    let value = if spec.log_scale && spec.min > 0.0 {
        let log_min = spec.min.ln();
        let log_max = spec.max.ln();
        (log_min + rng.gen_range(0.0..1.0) * (log_max - log_min)).exp()
    } else {
        spec.min + rng.gen_range(0.0..1.0) * (spec.max - spec.min)
    };
    scirs2_core::numeric::NumCast::from(value).unwrap_or_else(T::zero)
}

/// Sample a fresh, genuinely random architecture: a random number of components
/// drawn (with replacement) from [`OPTIMIZER_COMPONENT_VOCAB`], every
/// hyperparameter in [`HYPERPARAMETER_SPECS`] drawn from its own range, and a
/// chain of connections linking consecutive components.
///
/// `id_prefix` distinguishes the origin of the architecture in logs and in the
/// Pareto front's solution metadata.
pub fn sample_architecture<T: Float + Debug + Send + Sync + 'static>(
    rng: &mut Random<scirs2_core::random::rngs::StdRng>,
    id_prefix: &str,
) -> OptimizerArchitecture<T> {
    let component_count =
        rng.gen_range(MIN_SAMPLED_COMPONENTS..=MAX_SAMPLED_COMPONENTS.max(MIN_SAMPLED_COMPONENTS));
    let components: Vec<String> = (0..component_count)
        .map(|_| {
            OPTIMIZER_COMPONENT_VOCAB[rng.gen_range(0..OPTIMIZER_COMPONENT_VOCAB.len())].to_string()
        })
        .collect();

    let mut parameters = HashMap::new();
    for spec in HYPERPARAMETER_SPECS.iter() {
        parameters.insert(spec.name.to_string(), sample_hyperparameter::<T>(spec, rng));
    }

    let connections: Vec<(usize, usize)> = (1..components.len()).map(|i| (i - 1, i)).collect();

    OptimizerArchitecture {
        components,
        parameters: parameters.clone(),
        connections,
        metadata: HashMap::new(),
        hyperparameters: parameters,
        architecture_id: format!("{}_{:016x}", id_prefix, rng.gen_range(0..u64::MAX)),
    }
}

/// Uniform crossover over **both** the component sequence and the numeric
/// hyperparameters.
///
/// * Component length is inherited from a randomly chosen parent, and each
///   position independently takes the component from either parent (falling back
///   to the parent that has that position when lengths differ). This is what
///   lets the search recombine *structure*, which the previous
///   parameter-only crossover could not do.
/// * Each numeric parameter is taken from one parent or the other with equal
///   probability; `parameters` and `hyperparameters` are kept in agreement.
/// * Connections are rebuilt as a chain over the resulting component count, so
///   the offspring never carries an index that points past its own components.
pub fn crossover_architectures<T: Float + Debug + Send + Sync + 'static>(
    rng: &mut Random<scirs2_core::random::rngs::StdRng>,
    parent1: &OptimizerArchitecture<T>,
    parent2: &OptimizerArchitecture<T>,
) -> OptimizerArchitecture<T> {
    let target_len = if rng.gen_range(0.0..1.0) < 0.5 {
        parent1.components.len()
    } else {
        parent2.components.len()
    }
    .max(1);

    let mut components = Vec::with_capacity(target_len);
    for position in 0..target_len {
        let from_first = rng.gen_range(0.0..1.0) < 0.5;
        let primary = if from_first { parent1 } else { parent2 };
        let secondary = if from_first { parent2 } else { parent1 };
        let component = primary
            .components
            .get(position)
            .or_else(|| secondary.components.get(position))
            .cloned()
            .unwrap_or_else(|| {
                OPTIMIZER_COMPONENT_VOCAB[rng.gen_range(0..OPTIMIZER_COMPONENT_VOCAB.len())]
                    .to_string()
            });
        components.push(component);
    }

    let mut parameters: HashMap<String, T> = HashMap::new();
    let mut keys: Vec<&String> = parent1.parameters.keys().collect();
    for key in parent2.parameters.keys() {
        if !parent1.parameters.contains_key(key) {
            keys.push(key);
        }
    }
    keys.sort();
    for key in keys {
        let take_first = rng.gen_range(0.0..1.0) < 0.5;
        let chosen = if take_first {
            parent1
                .parameters
                .get(key)
                .or_else(|| parent2.parameters.get(key))
        } else {
            parent2
                .parameters
                .get(key)
                .or_else(|| parent1.parameters.get(key))
        };
        if let Some(value) = chosen {
            parameters.insert(key.clone(), *value);
        }
    }

    let connections: Vec<(usize, usize)> = (1..components.len()).map(|i| (i - 1, i)).collect();

    OptimizerArchitecture {
        components,
        parameters: parameters.clone(),
        connections,
        metadata: HashMap::new(),
        hyperparameters: parameters,
        architecture_id: format!("offspring_{:016x}", rng.gen_range(0..u64::MAX)),
    }
}

/// Mutate an architecture in place.
///
/// With probability `component_rate` **per component position** the component is
/// replaced by another draw from the vocabulary, and with the same probability
/// the architecture may grow or shrink by one position (staying within
/// `[MIN_SAMPLED_COMPONENTS, MAX_SAMPLED_COMPONENTS]`). With probability
/// `parameter_rate` each numeric parameter receives Gaussian-like jitter of
/// `+/-10%` of its known range, clamped back into that range when the name is a
/// known hyperparameter.
///
/// Returns `true` when anything actually changed, which lets callers record an
/// honest [`super::core::CreationMethod`].
pub fn mutate_architecture<T: Float + Debug + Send + Sync + 'static>(
    rng: &mut Random<scirs2_core::random::rngs::StdRng>,
    architecture: &mut OptimizerArchitecture<T>,
    component_rate: f64,
    parameter_rate: f64,
) -> bool {
    let mut changed = false;

    for position in 0..architecture.components.len() {
        if rng.gen_range(0.0..1.0) < component_rate {
            let replacement =
                OPTIMIZER_COMPONENT_VOCAB[rng.gen_range(0..OPTIMIZER_COMPONENT_VOCAB.len())];
            if architecture.components[position] != replacement {
                architecture.components[position] = replacement.to_string();
                changed = true;
            }
        }
    }

    // Structural growth / shrinkage.
    if rng.gen_range(0.0..1.0) < component_rate {
        if rng.gen_range(0.0..1.0) < 0.5 && architecture.components.len() > MIN_SAMPLED_COMPONENTS {
            let victim = rng.gen_range(0..architecture.components.len());
            architecture.components.remove(victim);
            changed = true;
        } else if architecture.components.len() < MAX_SAMPLED_COMPONENTS {
            architecture.components.push(
                OPTIMIZER_COMPONENT_VOCAB[rng.gen_range(0..OPTIMIZER_COMPONENT_VOCAB.len())]
                    .to_string(),
            );
            changed = true;
        }
    }

    let mut parameter_names: Vec<String> = architecture.parameters.keys().cloned().collect();
    parameter_names.sort();
    for name in parameter_names {
        if rng.gen_range(0.0..1.0) >= parameter_rate {
            continue;
        }
        let Some(current) = architecture.parameters.get(&name).copied() else {
            continue;
        };
        let current_f64 = current.to_f64().unwrap_or(0.0);
        let (lower, upper) = hyperparameter_bounds(&name).unwrap_or_else(|| {
            // Unknown parameter: jitter relative to its own magnitude and leave
            // it unbounded rather than inventing a range for it.
            let magnitude = current_f64.abs().max(1e-8);
            (current_f64 - magnitude, current_f64 + magnitude)
        });
        let step = (upper - lower) * 0.1;
        let jitter = (rng.gen_range(0.0..1.0) * 2.0 - 1.0) * step;
        let mutated = (current_f64 + jitter).clamp(lower.min(upper), upper.max(lower));
        if (mutated - current_f64).abs() > 0.0 {
            architecture.parameters.insert(
                name.clone(),
                scirs2_core::numeric::NumCast::from(mutated).unwrap_or(current),
            );
            changed = true;
        }
    }

    // Keep the derived fields consistent with the mutated component list.
    architecture.connections = (1..architecture.components.len())
        .map(|i| (i - 1, i))
        .collect();
    architecture.hyperparameters = architecture.parameters.clone();
    if changed {
        architecture.architecture_id = format!("mutant_{:016x}", rng.gen_range(0..u64::MAX));
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> Random<scirs2_core::random::rngs::StdRng> {
        Random::seed(seed)
    }

    #[test]
    fn sampling_produces_structurally_diverse_architectures() {
        let mut rng = rng(7);
        let population: Vec<OptimizerArchitecture<f64>> = (0..40)
            .map(|_| sample_architecture(&mut rng, "ind"))
            .collect();

        let distinct_signatures: std::collections::HashSet<Vec<String>> = population
            .iter()
            .map(|arch| arch.components.clone())
            .collect();
        assert!(
            distinct_signatures.len() > 5,
            "expected structurally diverse components, got {} distinct signatures",
            distinct_signatures.len()
        );

        // The pre-fix sampler returned exactly ["Adam"] every single time.
        assert!(
            distinct_signatures.len() > 1
                && !distinct_signatures
                    .iter()
                    .all(|sig| sig == &vec!["Adam".to_string()]),
            "the sampler must not collapse to a single repeated architecture"
        );

        // Learning rates must actually differ, and stay inside their range.
        let rates: std::collections::HashSet<u64> = population
            .iter()
            .filter_map(|arch| arch.parameters.get("learning_rate"))
            .map(|v| v.to_bits())
            .collect();
        assert!(
            rates.len() > 5,
            "expected varied learning rates, got {}",
            rates.len()
        );
        for arch in &population {
            let lr = arch.parameters["learning_rate"];
            assert!(
                (1e-5..=1e-1).contains(&lr),
                "learning rate {lr} out of range"
            );
            assert!(!arch.components.is_empty());
            assert_eq!(
                arch.connections.len(),
                arch.components.len().saturating_sub(1)
            );
            assert_eq!(arch.parameters, arch.hyperparameters);
        }
    }

    #[test]
    fn sampling_is_reproducible_for_a_given_seed() {
        let mut a = rng(99);
        let mut b = rng(99);
        let first: OptimizerArchitecture<f64> = sample_architecture(&mut a, "x");
        let second: OptimizerArchitecture<f64> = sample_architecture(&mut b, "x");
        assert_eq!(first.components, second.components);
        assert_eq!(first.parameters, second.parameters);
    }

    #[test]
    fn crossover_recombines_components_not_only_parameters() {
        let mut rng = rng(11);
        let parent1 = OptimizerArchitecture::<f64> {
            components: vec!["SGD".to_string(), "SGD".to_string(), "SGD".to_string()],
            parameters: [("learning_rate".to_string(), 0.001)].into_iter().collect(),
            connections: vec![(0, 1), (1, 2)],
            metadata: HashMap::new(),
            hyperparameters: [("learning_rate".to_string(), 0.001)].into_iter().collect(),
            architecture_id: "p1".to_string(),
        };
        let parent2 = OptimizerArchitecture::<f64> {
            components: vec!["Lion".to_string(), "Lion".to_string(), "Lion".to_string()],
            parameters: [("learning_rate".to_string(), 0.05)].into_iter().collect(),
            connections: vec![(0, 1), (1, 2)],
            metadata: HashMap::new(),
            hyperparameters: [("learning_rate".to_string(), 0.05)].into_iter().collect(),
            architecture_id: "p2".to_string(),
        };

        // Over many draws at least one offspring must mix the two vocabularies,
        // which the previous parameter-only crossover could never produce.
        let mut saw_mixed = false;
        for _ in 0..50 {
            let child = crossover_architectures(&mut rng, &parent1, &parent2);
            let has_sgd = child.components.iter().any(|c| c == "SGD");
            let has_lion = child.components.iter().any(|c| c == "Lion");
            assert!(
                child.components.iter().all(|c| c == "SGD" || c == "Lion"),
                "offspring components must come from the parents: {:?}",
                child.components
            );
            if has_sgd && has_lion {
                saw_mixed = true;
            }
            assert_eq!(
                child.connections.len(),
                child.components.len().saturating_sub(1)
            );
        }
        assert!(
            saw_mixed,
            "crossover never recombined the component sequences"
        );
    }

    #[test]
    fn crossover_handles_parents_of_different_length() {
        let mut rng = rng(3);
        let short = OptimizerArchitecture::<f64> {
            components: vec!["Adam".to_string()],
            parameters: HashMap::new(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters: HashMap::new(),
            architecture_id: "s".to_string(),
        };
        let long = OptimizerArchitecture::<f64> {
            components: vec![
                "SGD".to_string(),
                "RMSprop".to_string(),
                "Momentum".to_string(),
                "Lion".to_string(),
            ],
            parameters: HashMap::new(),
            connections: vec![(0, 1), (1, 2), (2, 3)],
            metadata: HashMap::new(),
            hyperparameters: HashMap::new(),
            architecture_id: "l".to_string(),
        };
        for _ in 0..30 {
            let child = crossover_architectures(&mut rng, &short, &long);
            assert!(!child.components.is_empty());
            assert!(child.components.len() <= 4);
            for (from, to) in &child.connections {
                assert!(*from < child.components.len());
                assert!(*to < child.components.len());
            }
        }
    }

    #[test]
    fn mutation_changes_components_and_respects_bounds() {
        let mut rng = rng(23);
        let mut architecture: OptimizerArchitecture<f64> = sample_architecture(&mut rng, "seed");
        let original_components = architecture.components.clone();

        let mut component_changed = false;
        for _ in 0..40 {
            let mut candidate = architecture.clone();
            if mutate_architecture(&mut rng, &mut candidate, 1.0, 1.0)
                && candidate.components != original_components
            {
                component_changed = true;
            }
            // Bounds must hold after every mutation.
            for (name, value) in &candidate.parameters {
                if let Some((lower, upper)) = hyperparameter_bounds(name) {
                    assert!(
                        *value >= lower - 1e-12 && *value <= upper + 1e-12,
                        "{name} = {value} escaped [{lower}, {upper}]"
                    );
                }
            }
            assert!(candidate.components.len() >= MIN_SAMPLED_COMPONENTS);
            assert!(candidate.components.len() <= MAX_SAMPLED_COMPONENTS);
            assert_eq!(candidate.parameters, candidate.hyperparameters);
            architecture = candidate;
        }
        assert!(
            component_changed,
            "mutation never touched the component sequence"
        );
    }

    #[test]
    fn mutation_with_zero_rates_is_a_no_op() {
        let mut rng = rng(5);
        let architecture: OptimizerArchitecture<f64> = sample_architecture(&mut rng, "seed");
        let mut candidate = architecture.clone();
        assert!(!mutate_architecture(&mut rng, &mut candidate, 0.0, 0.0));
        assert_eq!(candidate.components, architecture.components);
        assert_eq!(candidate.parameters, architecture.parameters);
    }

    #[test]
    fn hyperparameter_metadata_is_consistent() {
        for name in hyperparameter_names() {
            let (lower, upper) = hyperparameter_bounds(name).expect("known bounds");
            assert!(lower < upper, "{name} has an empty range");
        }
        assert!(hyperparameter_bounds("not_a_hyperparameter").is_none());
    }
}
