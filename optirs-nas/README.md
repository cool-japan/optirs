# OptiRS NAS

Neural architecture search and hyperparameter optimization for the OptiRS machine-learning
optimization library.

## Scope

`optirs-nas` searches for **optimizer architectures** — which optimizer components to compose
(SGD, Adam, AdamW, RMSprop, AdaGrad, Momentum, Lion, schedulers, regularizers) and with which
hyperparameters — and provides the hyperparameter-search machinery around that. It is not a
computer-vision layer-topology search: the search space is
[`nas_engine::SearchSpaceConfig`], whose unit is an optimizer component.

Status: **research-grade**. Every algorithm listed below is implemented and tested in this crate,
but the search-space and objective APIs may still change between releases. Validate a discovered
configuration against your own baselines before relying on it.

> Every Rust block in this file is compiled and run by `cargo test --doc` (see
> `ReadmeDoctests` in `src/lib.rs`), so the examples cannot drift away from the API.

## Dependencies

```toml
[dependencies]
optirs-nas = "0.3.2"
```

`scirs2-core` is pulled in transitively; you do not need to depend on it yourself unless you use
its types directly. The crate has **no intra-workspace dependencies** — it builds and tests
standalone.

There are **no Cargo features**. Earlier versions declared `bayesian`, `evolutionary`,
`reinforcement`, `progressive`, `multi_objective` and `learned_integration`; none of them gated
any code (there was not a single `cfg(feature = ...)` in the crate), so they have been removed
rather than left as documentation that promised a build knob that did not exist.

## What is implemented

| Area | Module | Notes |
|---|---|---|
| Random search | [`search_strategies::random`] | OS-entropy seeded, or reproducible with an explicit seed |
| Evolutionary search | [`search_strategies::evolutionary`] | population-based, real crossover/mutation |
| Bayesian optimization | [`search_strategies::bayesian`] | Gaussian-process surrogate with EI/PI/UCB |
| RL controller | [`search_strategies::rl_search`] | LSTM controller trained by REINFORCE |
| Differentiable (DARTS) | [`search_strategies::differentiable`] | plus PC-DARTS-style and robust variants |
| Neural predictor | [`search_strategies::neural_predictor`] | trained by real backpropagation; MC-dropout uncertainty |
| Progressive search | [`search_strategies::progressive`] | staged complexity growth |
| Multi-objective | [`multi_objective`] | NSGA-II and weighted-sum, exact hypervolume |
| Hyperparameter search | [`hyperparameter`] | grid enumeration, TPE, GP-free surrogate, evolutionary |
| Hardware cost models | [`hardware_cost`] | latency / memory / energy estimation |
| Domain search spaces | [`domain_specific_nas`] | CV, NLP, TimeSeries, Reinforcement, Scientific |
| Architecture embedding | [`architecture_embedding`] | vector-space similarity over architectures |
| AutoML coordination | [`automl_pipeline`] | pipeline stages around a search |

Multi-objective algorithms **other than** NSGA-II and weighted-sum (NSGA-III, MOEA/D, PAES,
SPEA2, epsilon-constraint, goal programming) are *not* implemented. Configuring one returns
`OptimError::NotImplemented` at engine construction instead of silently substituting NSGA-II or
producing an empty Pareto front.

## Running a search

```rust
use optirs_nas::nas_engine::resources::SystemResourceTracker;
use optirs_nas::nas_engine::telemetry::{FixedTelemetry, TelemetrySample};
use optirs_nas::nas_engine::{
    create_minimal_nas_config, NeuralArchitectureSearch, SearchStrategyType,
};
use std::time::Duration;

# fn main() -> Result<(), optirs_nas::error::OptimError> {
let mut config = create_minimal_nas_config::<f64>();
config.search_strategy = SearchStrategyType::Evolutionary;
config.search_budget = 2;
config.population_size = 4;

let mut engine = NeuralArchitectureSearch::new(config)?;

// The engine reports which strategy is actually running, not just what was requested.
assert_eq!(engine.search_strategy_name(), "EvolutionaryStrategy");

// Pin the telemetry so this example does not depend on the host's free memory.
// `resource_monitor_mut` is also the supported injection point for real telemetry.
engine.resource_monitor_mut().set_trackers(vec![Box::new(
    SystemResourceTracker::with_telemetry(
        "example".to_string(),
        Duration::from_secs(5),
        Box::new(FixedTelemetry::new("example", TelemetrySample::unknown())),
    ),
)]);

let results = engine.run_search()?;
assert!(!results.search_history.is_empty());
# Ok(())
# }
```

Every `SearchStrategyType` resolves to a real implementation. There is no
"falls back to Evolutionary" arm.

## Hyperparameter search

```rust
use optirs_nas::hyperparameter::{
    DistributionType, HyperparameterOptimizer, HyperparameterSpace, OptimizationStrategy,
    ParameterRange,
};

# fn main() -> Result<(), optirs_nas::error::OptimError> {
let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
space.add_parameter(
    "learning_rate".to_string(),
    ParameterRange {
        name: "learning_rate".to_string(),
        min_value: 1e-5,
        max_value: 1e-1,
        distribution: DistributionType::Uniform,
        log_scale: true,
        discrete_values: None,
    },
);
space.add_categorical_parameter(
    "optimizer".to_string(),
    vec!["adam".to_string(), "sgd".to_string()],
);

// Grid search really enumerates the grid: 4 log-spaced rates x 2 optimizers.
let mut optimizer =
    HyperparameterOptimizer::with_seed(space, OptimizationStrategy::Grid, 7);
optimizer.set_grid_resolution(4);
assert_eq!(optimizer.grid_size(), Some(8));

let suggestion = optimizer.suggest_configuration()?;
let learning_rate = suggestion.parameters["learning_rate"];
assert!((1e-5..=1e-1).contains(&learning_rate));
# Ok(())
# }
```

Available strategies and what they do:

- `Random` — samples each parameter from its declared `DistributionType` (all five are
  implemented; none silently degrades to uniform).
- `Grid` — deterministic mixed-radix enumeration; the *n*-th call returns the *n*-th grid point
  and the whole grid is covered before it wraps. Continuous axes are log-spaced when the range
  declares `log_scale`.
- `TPE` — the published Tree-structured Parzen Estimator: the history is split by score
  quantile and **two** adaptive kernel-density estimators are fitted, candidates are drawn from
  the "good" density and ranked by the density ratio.
- `Bayesian` — a GP-free Nadaraya-Watson kernel-regression surrogate whose predictive variance
  grows away from the data, maximized under Expected Improvement / Probability of Improvement /
  Upper Confidence Bound / an entropy term.
- `Evolutionary` — tournament selection over a bounded elitist population, recombination per
  parameter, and Gaussian mutation on the parameter's own (log-aware) axis.

`TPE`, `Bayesian` and `Evolutionary` require a random *initial design* before they can be
fitted — that is part of the published algorithms. During that phase
`last_strategy_used()` reports `Random`, so the phase is visible rather than hidden:

```rust
use optirs_nas::hyperparameter::{
    DistributionType, HyperparameterOptimizer, HyperparameterSpace, OptimizationStrategy,
    ParameterRange,
};

# fn main() -> Result<(), optirs_nas::error::OptimError> {
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

let mut optimizer =
    HyperparameterOptimizer::with_seed(space, OptimizationStrategy::TPE, 5);
let _first = optimizer.suggest_configuration()?;
assert_eq!(optimizer.last_strategy_used(), OptimizationStrategy::Random);
# Ok(())
# }
```

`ParticleSwarm`, `SuccessiveHalving`, `Hyperband` and `BOHB` are declared in the enum but not
implemented; selecting one returns `OptimError::NotImplemented`.

## Multi-objective search and the hypervolume indicator

The hypervolume is the **exact** indicator (recursive slicing / HSO), not a bounding-box
heuristic, and it handles mixed `Minimize` / `Maximize` directions:

```rust
use optirs_nas::multi_objective::hypervolume_minimization;

// Union of [1,2]x[0,2] and [0,2]x[1,2] against the reference point (2, 2).
let front = vec![vec![1.0_f64, 0.0], vec![0.0, 1.0]];
let hypervolume = hypervolume_minimization(&front, &[2.0, 2.0]);
assert!((hypervolume - 3.0).abs() < 1e-12);

// A dominated point adds nothing; a genuinely non-dominated one strictly increases it.
let with_dominated = vec![vec![1.0_f64, 0.0], vec![0.0, 1.0], vec![1.5, 1.5]];
assert!((hypervolume_minimization(&with_dominated, &[2.0, 2.0]) - 3.0).abs() < 1e-12);
let with_new = vec![vec![1.0_f64, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
assert!(hypervolume_minimization(&with_new, &[2.0, 2.0]) > 3.0);
```

`HypervolumeCalculator` wraps this with caching and a method selector. `WFG`, `Quick` and `HSO`
are all exact and return identical values (documented on the enum); `MonteCarlo` is a real,
seeded box-sampling estimator, not a hand-off to the exact routine.

Front quality metrics — `convergence`, `objective_space_coverage`, `reference_distance`,
`epsilon_dominance`, `spread`, `spacing` — are all computed from the front
([`multi_objective::metrics`]). `convergence` is the relative change of the hypervolume since
the previous update, measured against a reference point that is latched on first use so
generations stay comparable.

## Resource monitoring

Resource telemetry is **injectable**, and the default source reports only what it can honestly
measure — `std::thread::available_parallelism` everywhere, plus `/proc` on Linux — and `None`
for everything else (GPU count, disk capacity, network bandwidth, temperature, power). An
unmeasurable resource is treated as *unlimited*, so a resource constraint can never abort a
search on the strength of a value nobody measured:

```rust
use optirs_nas::nas_engine::telemetry::{StdTelemetry, TelemetrySource};

let sample = StdTelemetry::new().sample();

// Genuinely measurable.
assert!(sample.logical_cpus.is_some());

// Not reachable in pure Rust: reported as unknown, never invented.
assert!(sample.gpu_devices.is_none());
assert!(sample.temperature_celsius.is_none());
assert!(sample.power_watts.is_none());
```

To enforce GPU / thermal / power budgets, supply your own `TelemetrySource` and install it with
`NeuralArchitectureSearch::resource_monitor_mut().set_trackers(..)` +
`SystemResourceTracker::with_telemetry(..)`.

Resource enforcement lives in exactly one place: the search loop samples the monitor each
generation and returns `OptimError::ResourceLimitExceeded` on a violation. It never truncates a
search silently.

## Domain-specific search spaces

```rust
use optirs_nas::domain_specific_nas::{DomainNASEngine, DomainType};

# fn main() -> Result<(), optirs_nas::error::OptimError> {
let mut engine = DomainNASEngine::<f64>::new_for_domain(DomainType::ComputerVision);

// Search within the domain's pre-configured space, then check the winner against
// the domain's own structural rules.
let architectures = engine.search(4)?;
assert!(!architectures.is_empty());

if let Some(best) = engine.get_best_architecture() {
    let issues = engine.validate_for_domain(best)?;
    // `issues` is a (possibly empty) list of human-readable rule violations.
    assert!(issues.len() < 100);
}
# Ok(())
# }
```

Supported domains ([`domain_specific_nas::DomainType`]): `ComputerVision`,
`NaturalLanguageProcessing`, `TimeSeries`, `Reinforcement`, `Scientific`, `Multimodal`.
Constraints are expressed with [`domain_specific_nas::NASConstraint`]
(`MaxLatencyMs`, `MaxMemoryMb`, `MinAccuracy`, `RequiresComponent`, `MaxDepth`, `MaxWidth`).

## Notes on API shape

This crate is entirely **synchronous**. It contains no `async fn`, so nothing here is `.await`ed.
Earlier revisions of this document showed `await` on every example; that was never valid.

Longer, runnable programs live in `examples/` rather than in this file.

## References

Techniques implemented here follow:

- Zoph & Le, *Neural Architecture Search with Reinforcement Learning* (2017)
- Liu et al., *DARTS: Differentiable Architecture Search* (2019)
- Deb et al., *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II* (2002)
- Bergstra, Bardenet, Bengio & Kegl, *Algorithms for Hyper-Parameter Optimization* (2011) — TPE
- While, Bradstreet & Barone, *A Fast Way of Calculating Exact Hypervolumes* (2012) — the exact
  slicing recursion

## Contributing

OptiRS follows the Cool Japan organization's development standards. See the main OptiRS
repository for contribution guidelines.

## License

Apache-2.0
