# OptiRS NAS TODO (v0.3.2)

## Module status

**Tests**: 519 passing, 0 ignored (`cargo nextest run -p optirs-nas --all-features`) + 16
passing doctests (`cargo test --doc -p optirs-nas --all-features`), 6 of which compile and
run every Rust code block in `README.md` (see `ReadmeDoctests` in `src/lib.rs` — the README
cannot drift away from the API without a doctest failure).
**Compiler warnings**: none — `cargo check`/`cargo clippy --all-features --all-targets` and
`cargo doc --all-features --no-deps` are all clean. No `#[allow(...)]` of any kind anywhere
in the crate.
**Dependencies**: none inside the workspace. `optirs-core` and `optirs-learned` were declared
once but never referenced, and were removed so this crate builds and tests standalone.
**Cargo features**: none. Earlier versions declared `bayesian`, `evolutionary`,
`reinforcement`, `progressive`, `multi_objective` and `learned_integration`; none of them
gated a single line of code, so they were removed rather than left as documentation for a
build knob that did not exist.

This file describes what is actually implemented and tested, not an aspirational feature
list. See `README.md` for usage and the module docs in `src/lib.rs` for the authoritative
implementation-status summary.

---

## Done: search strategies (`search_strategies`)

- [x] Random search (`random`) — OS-entropy seeded by default, or reproducible with an
      explicit seed.
- [x] Evolutionary search (`evolutionary`) — tournament selection over a bounded elitist
      population, real crossover/mutation, id-keyed fitness lookup (not positional).
- [x] Bayesian optimization (`bayesian`) — Gaussian-process surrogate with
      EI/PI/UCB acquisition.
- [x] RL controller (`rl_search`) — LSTM controller trained by REINFORCE, with a learned
      baseline, an entropy bonus and gradient clipping.
- [x] Differentiable search / DARTS (`differentiable`) — plus a PC-DARTS-style
      `MemoryEfficientDARTS` (deterministic top-k channel selection, not the published
      random subset — documented at `partial_channel_mask`) and `RobustDARTS`.
- [x] Neural predictor (`neural_predictor`) — trained by real backpropagation through cached
      pre-activations; MC-dropout uncertainty gates exploit-vs-explore candidate selection.
- [x] Progressive search (`progressive`) — staged complexity growth, id-keyed result
      attribution (a result is credited to the phase that generated it, never to whichever
      architecture happened to be generated last), `is_search_complete()` completion signal
      consulted by the engine's stop condition.

Every `SearchStrategyType` resolves to its own real adapter; none of them silently falls back
to `Evolutionary` or `Random`.

## Done: multi-objective optimization (`multi_objective`)

- [x] NSGA-II — non-dominated sorting, crowding distance, real tournament/crossover/mutation.
- [x] NSGA-III (Deb & Jain, 2014) — reference-direction niching; reuses NSGA-II's dominance
      sort by composition rather than a second implementation.
- [x] MOEA/D (Zhang & Li, 2007) — Das-Dennis lattice decomposition, `B(i)` neighbourhood
      mating, Tchebycheff / PBI / ASF / weighted-sum scalarized replacement, external archive.
- [x] Weighted-sum scalarization.
- [x] Exact hypervolume (recursive slicing / HSO), not a bounding-box heuristic; `WFG`,
      `Quick` and `HSO` all return identical values, `MonteCarlo` is a real seeded
      box-sampling estimator.
- [x] Front-quality metrics shared by all four optimizers: `convergence`,
      `objective_space_coverage`, `reference_distance`, `epsilon_dominance`, `spread`,
      `spacing`.

Declared but not implemented — `MultiObjectiveAlgorithm::{SPEA2, PAES, EpsilonConstraint,
GoalProgramming, Custom(_)}` return `OptimError::NotImplemented` at engine construction,
never a silent substitution of NSGA-II or an empty Pareto front.

`MultiObjectiveConfig::user_preferences` / `::constraint_handling` are declared and honestly
documented as not yet consulted by any optimizer (every optimizer treats all candidates as
feasible). Hard resource limits are a separate mechanism — see Resource monitoring below.

## Done: hyperparameter search (`hyperparameter`)

- [x] `Random` — samples each parameter from its declared `DistributionType`; all five
      (`Uniform`, `Normal`, `LogNormal`, `Beta`, `Exponential`) are implemented.
- [x] `Grid` — deterministic mixed-radix enumeration; log-spaced continuous axes when the
      range declares `log_scale`.
- [x] `TPE` — the published Tree-structured Parzen Estimator: history split by score
      quantile, two adaptive kernel-density estimators, candidates ranked by density ratio.
- [x] `Bayesian` — a GP-free Nadaraya-Watson kernel-regression surrogate maximized under
      EI/PI/UCB/an entropy term.
- [x] `Evolutionary` — tournament selection, per-parameter recombination, Gaussian mutation
      on each parameter's own (log-aware) axis.

Declared but not implemented — `OptimizationStrategy::{ParticleSwarm, SuccessiveHalving,
Hyperband, BOHB}` return `OptimError::NotImplemented`.

## Done: everything else

- [x] Hardware cost models (`hardware_cost.rs`) — per-layer FLOPs/params/activation-memory,
      latency roofline (compute- vs bandwidth-bound), an energy model, `LatencyLookupTable`,
      and edge-CPU / mobile-GPU / server-GPU `HardwareProfile`s.
- [x] Domain-specific search spaces (`domain_specific_nas.rs`) — `ComputerVision`, `NLP`,
      `TimeSeries`, `Reinforcement`, `Scientific`, `Multimodal`, each with its own
      `NASConstraint` set (`MaxLatencyMs`, `MaxMemoryMb`, `MinAccuracy`, `RequiresComponent`,
      `MaxDepth`, `MaxWidth`).
- [x] Speech NAS (`speech_nas.rs`) — `SpeechNasEngine` over Mel-filterbank / Conv1D / LSTM /
      BiLSTM / Attention / CTC-decoder layers, Pareto front over (WER, latency, memory).
- [x] Multimodal NAS (`multimodal_nas/`) — per-modality encoders + fusion ops (EarlyConcat /
      LateFusion / CrossAttention / Gated / Bilinear), Pareto front over
      (accuracy, latency, memory).
- [x] Architecture embedding + similarity (`architecture_embedding.rs`).
- [x] Cross-domain transfer (`cross_domain_transfer.rs`) — transferability scoring,
      warm-start embeddings.
- [x] Few-shot architecture optimization (`few_shot_architecture.rs`) — Prototypical /
      Matching / MAML-adapter / distance-weighted-KNN.
- [x] Architecture knowledge graph (`architecture_knowledge_graph.rs`) —
      random-walk-with-restart propagation, JSON persistence.
- [x] AutoML pipeline coordination (`automl_pipeline/`) — preprocessing, feature
      engineering, model selection and ensembling stages around a user-supplied evaluator.
- [x] Resource monitoring — `TelemetrySource` is injectable; the default reports only what
      it can honestly measure (`std::thread::available_parallelism`, plus `/proc` on Linux)
      and `None` for everything else (GPU count, disk, network, temperature, power). An
      unmeasurable resource reads as unlimited, never a fabricated number, and the search
      loop checks it once per generation, returning `OptimError::ResourceLimitExceeded` on a
      real violation rather than truncating a search silently.

---

## Out of scope (declared in the type system, not silently faked)

- Multi-objective: `SPEA2`, `PAES`, `EpsilonConstraint`, `GoalProgramming`, `Custom(_)`.
- Hyperparameter search: `ParticleSwarm`, `SuccessiveHalving`, `Hyperband`, `BOHB`.
- Pareto-level constraint handling and interactive preference articulation (see above).
- Real GPU / disk / network / thermal / power telemetry — needs FFI, which is off by default
  under COOLJAPAN policy. `TelemetrySource` accepts a caller-supplied real reading instead of
  fabricating one.

---

**Status**: Research-grade. Every algorithm above is implemented and tested, but the
search-space and objective APIs may still change between releases — validate a discovered
configuration against your own baselines before relying on it.
**Version**: v0.3.2
