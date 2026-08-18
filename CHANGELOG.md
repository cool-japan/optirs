# Changelog

All notable changes to OptiRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2026-08-18

A production-hardening release. The organizing theme is honesty: throughout the
workspace, code that *simulated* a result — fabricated p-values, hardcoded success
rates, `thread::sleep` standing in for device latency, secure-aggregation masks that
were never actually cancelled — was either replaced with a real implementation or
changed to return an explicit error naming what it cannot do. No path in this release
reports a number it did not compute.

No optimizer, scheduler or regularizer was removed or renamed, so ordinary training code
should need no changes. A large amount of never-implemented scaffolding *was* deleted,
though, so code that named those types will no longer compile — see **Removed**.

### Added

#### Streaming, drift detection and anomaly detection (`optirs-core::streaming`)

- **Real statistical drift tests.** A new shared numerics module
  (`streaming/adaptive_streaming/statistics.rs`) provides NaN-safe total ordering,
  in-place median/quantile selection, the standard-normal survival function (through
  the `scirs2-stats` Gaussian CDF), a Lanczos `ln_gamma` plus regularized upper
  incomplete gamma driving an accurate chi-square SF, the Kolmogorov distribution SF,
  the two-sample KS statistic, smoothed histograms/PMFs, KL / Jensen-Shannon /
  Hellinger divergences, 1-D Wasserstein distance and the G-test statistic. The drift
  tests, distribution comparators and linear-model detector in `drift_tests.rs` are
  built on these instead of on invented p-values.
- **Real ML anomaly detectors** (`anomaly_ml.rs`), with the z-score and IQR detectors
  separated into `anomaly_statistical.rs`. Detector success rates are measured, not
  hardcoded.
- **Drift-detector models** — `NeuralNetwork`, `DecisionTree` and `Ensemble`
  (`drift_models.rs`), each fitted against a **Winsorised prequential error baseline**
  so that a single large error can no longer inflate the baseline variance the detector
  is testing against.
- **Per-context drift-detector banks**: `ContextAwareDriftDetector` now builds and keeps
  an independent detector per observed context instead of sharing one global detector.
- **Full-curve AUC-ROC** (`anomaly_scoring.rs`): scores are retained in a bounded buffer
  (4096 entries) and integrated with a tie-grouped trapezoidal rule. Fewer than two
  labelled scores per class is an explicit error rather than a single-point guess.
- **Adaptive ensemble voting** (`anomaly_ensemble.rs`): detector weights are per-detector
  balanced accuracy measured from recorded outcomes, with a documented uniform fallback.
- **Time-bucketed metric aggregation** with retention eviction
  (`streaming/streaming_metrics/aggregation.rs`); the configured aggregated-retention
  window is now enforced instead of being read by nobody.
- **Contextual-bandit meta-learning** arms, feature standardisation and state/action
  encoding (`meta_bandit.rs`).
- `LowLatencyOptimizer` now owns its parameter vector, so `exact_update` returns an
  actual optimization trajectory rather than one step away from the origin.
- Resource management now tracks this process's own memory footprint separately from
  system-wide usage, so per-process allocation budgets are checked against a per-process
  figure.

#### Privacy (`optirs-core::privacy`)

- **Real Bonawitz secure aggregation** (`privacy/federated/`). Each client generates a
  per-round X25519 key pair and publishes only the public half; pairwise seeds come from
  a real ECDH shared secret hashed with SHA-256 under a domain separator, masks are
  expanded by a SHA-256 counter-mode PRG with rejection sampling (uniform over the group,
  no modulo bias), and each unordered client pair contributes `+m` once and `−m` once so
  the server's sum telescopes. The server holds no key material and cannot reconstruct
  any client's mask. Low-order public keys and self-pairing are rejected.
  New pure-Rust dependency: `x25519-dalek`.
- **Robust aggregation operators** (`privacy/federated/robust_ops.rs`): coordinate-wise
  median, trimmed mean, Krum, Multi-Krum, Bulyan and centered clipping.
- **Outlier tests** (`privacy/federated/outlier_tests.rs`): z-score, modified z-score,
  IQR, Grubbs and Chauvenet.
- **Differentially private hyperparameter selection** via a real exponential mechanism
  (`private_hyperparameter_optimization/selection.rs`), plus report-noisy-max with
  Gumbel noise, Laplace and Gaussian selection; unsupported mechanisms are refused with
  a pointer to the real primitive.
- **Private Bayesian optimization** with a pure-Rust Gaussian process (Cholesky, RBF
  kernel), Expected Improvement acquisition and an invertible parameter encoding —
  `suggest_next` no longer returns an empty configuration.
- **Verifiable audit trails** (`privacy/enhanced_audit/`): an RFC-6962-style Merkle tree
  with domain-separated leaf/node tags, canonical length-prefixed event encoding,
  HMAC-SHA256 pinned against RFC 4231 vectors, constant-time digest comparison, and
  verification that re-derives every leaf from the caller's events and names the first
  offending index.
- **Federated privacy configuration validation** (`federated_privacy/validation.rs`),
  invoked from the coordinator constructor, plus real subsampling amplification
  (`ε / ln(1 + q(e^ε − 1))`) replacing a fabricated factor.

#### Checkpointing and plugins (`optirs-core`)

- **`FileCheckpointStorage`** — a filesystem-backed `CheckpointStorage` implementation
  (`coordination/orchestration/checkpoint_manager/file_storage.rs`) using `oxicode` for
  the on-disk codec and its CRC32 checksum wrapper for corruption detection. This
  required `Serialize`/`Deserialize` derives across the 47 types reachable from
  `Checkpoint<T>`.
- **Real `plugin.toml` parsing** (`plugin/loader/manifest.rs`) using the `toml` crate,
  including genuine array-of-tables dependency and permission entries, overlaid onto the
  previous defaults. `PluginCache` now honours its configured limits.

#### Hardware-aware and lifelong optimization (`optirs-core`)

- **A real optimizer step path** for `hardware_aware::OptimizationState`
  (`hardware_aware/optimization_state.rs`): it owns a boxed `Optimizer` and a boxed
  `LearningRateScheduler`, applies the schedule's current rate before stepping and
  advances it after, and honours gradient accumulation.
- **Recommendation-driven optimizer selection**: `HardwareOptimizerKind::recommend_for`
  chooses between SGD / Lion / Adam / LAMB on a stated per-parameter optimizer-state
  footprint rationale, reported by `state_buffers_per_parameter()`.
- **Similarity-driven cross-task transfer** for `LifelongOptimizer`
  (`online_learning/transfer.rs`), with single-linkage task clustering at the transfer
  threshold and a stable hash for reproducible task grouping.
- **`AdaptiveTuner`** (`hardware_aware/adaptive_tuner.rs`) with working Grid, Greedy and
  genetic-algorithm search strategies. Bayesian and reinforcement-learning strategies
  return explicit errors rather than silently falling back to random search.

#### Neural architecture search (`optirs-nas`)

- **Exact hypervolume** by HSO recursion (`multi_objective/hypervolume.rs`), wired into
  NSGA-II against a latched reference point, with convergence and objective-space
  coverage actually computed.
- **NSGA-II** with real crossover and mutation over architecture components, a full
  non-dominated sort, and unevaluated individuals excluded from dominance.
- **MOEA/D** (Zhang & Li, 2007) with a Das-Dennis weight lattice
  (`multi_objective/moead.rs`) — previously every fallible method returned
  "not implemented".
- **Real hyperparameter search** (`hyperparameter/`): mixed-radix grid enumeration
  (`grid.rs`), a TPE implementation with two adaptive Parzen estimators (`tpe.rs`), and a
  kernel-regression surrogate with erf-based acquisition functions (`surrogate.rs`).
  These no longer delegate to random search.
- **Neural predictor** `backward_update` performs a real backward pass with input
  validation instead of returning `Ok(())`.

#### Learned optimizers (`optirs-learned`)

- **LSTM truncated BPTT meta-training** (`lstm/bptt.rs`, `lstm/trainer.rs`,
  `lstm/features.rs`), with gradients verified against finite differences (with and
  without attention) and meta-training verified to reduce held-out meta-loss.
- **Seeded, reproducible initialization**: `LSTMNetwork::new_seeded(config, seed)` threads
  a seeded RNG through the LSTM layers, output projection and attention mechanism;
  `new()` delegates with an entropy seed.
- **Transformer performance predictor** (`adaptive/performance_predictor.rs`): a
  Xavier-initialised random-feature map with ridge-fitted heads. An untrained predictor
  reports "no information" instead of the previous fabricated 0.15 / 0.92 / 0.85
  constants.
- **Real backward pass** in `transformer_based_optimizer`, propagating through the output
  projection, layer norms, feed-forward blocks and input embedding.
- **Distinct higher-order differentiation modes**: `HvpMode::{CentralDifference,
  ForwardDifference, QuadraticSecant, MaterializedHessian}` are four genuinely different
  algorithms (they were previously byte-identical function bodies);
  `HvpMode::NestedAutodiff` and `MixedPartialMethod::NestedAutodiff` return explicit
  errors explaining that nested AD cannot exist behind the engine's black-box `Fn`
  objective signature.
- **Evolution-strategies meta-training for `GnnOptimizer` and `NtmOptimizer`**
  (`gnn_optimizer/meta_training.rs`, `ntm_optimizer/meta_training.rs`), implementing the
  crate's `MetaTrainable` trait to flatten each architecture's learned weights into one
  vector, load them back, and clear per-rollout state between generations. Both
  optimizers previously drew their weights once from the configured seed and never
  updated them again, so every "learned" step ran on a permanently random network; they
  now actually train.

#### Benchmarking (`optirs-bench`)

- **Real cross-platform test execution** — the orchestrator's `execute_*_test` paths now
  run `docker exec`, a local `Command`, or `ssh`, and return an explicit error when the
  runtime is absent instead of reporting a fabricated pass.
- **True parallel test execution** (via `futures`), and a new `leak_tool_reports` module
  for parsing external leak-detector output.
- Real git metadata collection in the CI/CD automation path.

#### WebAssembly (`optirs-wasm`)

- `WasmGpuOptimizer` gained a real `wasm_bindgen` constructor, genuine `navigator.gpu`
  presence detection, and a real `requestAdapter()` → `requestDevice()` handshake that
  reads the actual adapter's vendor/architecture/description, returning an explicit error
  when WebGPU is unavailable. (Running WGSL compute kernels remains explicitly documented
  as not implemented.)

#### Tooling and policy

- **`deny.toml`** at the workspace root, enforced by `cargo deny check bans` across the
  Linux / macOS / Windows / `wasm32` targets. Banned in favour of the COOLJAPAN pure-Rust
  equivalents: BLAS/LAPACK FFI (`openblas-src`, `blas-src`, `lapack-src`, `intel-mkl-src`,
  `netlib-src`), `bincode`, the C regex/tokenizer toolchains reached through
  `onig`/`onig_sys`/`esaxx-rs`, `z3`, `rusqlite`, the compression family
  (`zip`/`flate2`/`zstd`/`bzip2`/`lz4`/`tar`/`snap`/`brotli`/`miniz_oxide`) and TLS/crypto
  FFI (`openssl`, `native-tls`, `ring`, `aws-lc-sys`). Wildcard version requirements are
  denied.

### Changed

- **Zero-warning workspace policy, with no blanket allows.** The workspace-level
  `[workspace.lints.rust]` table is now intentionally empty and every crate opts in via
  `[lints] workspace = true`; the previous blanket `#![allow(...)]` attributes were
  deleted from every crate. This exposed roughly 1,241 `rustc` and 1,360 `clippy`
  warnings that the allows had been hiding — unused imports, unused variables and dead
  code — all of which were fixed rather than re-suppressed. `cargo check` and
  `cargo clippy --workspace --all-features --all-targets` are both at zero warnings.
- **`unwrap`/`expect` sweep.** Production code no longer panics on numeric conversion or
  lock poisoning: `A::from(x).expect("unwrap failed")` call sites were replaced with
  fallible `cast_scalar` / `scalar_to_f64` helpers returning `OptimError`, and mutex
  handling recovers from poisoning instead of unwrapping. Remaining `expect` calls in
  test code were given messages that say what actually failed.
- **File-size policy.** Every source file in the workspace is now under 2,000 lines. The
  17 files that exceeded it were split into roughly 90 module files (largest result: 1,176
  lines), with public APIs preserved through re-exports; item inventories, `#[test]` counts
  and doc-comment counts were diffed against the originals in both directions to confirm
  that nothing was lost or duplicated. Files split include
  `optirs-core/src/privacy/byzantine_tolerance.rs`,
  `privacy/secure_multiparty.rs`, `coordination/orchestration/checkpoint_manager.rs`,
  `coordination/monitoring/anomaly_detection.rs`, `reinforcement_learning/actor_critic.rs`,
  `plugin/loader.rs`, `streaming/types.rs`, `optirs-nas/src/multi_objective.rs` and
  `optirs-tpu/src/tpu_backend.rs`.
- **GPU vendor backends are disclosed as host-memory simulations.** The
  `cuda`/`rocm`/`oneapi`/`metal` memory backends no longer sleep to imitate device
  latency, and each file's header now states plainly that its copy functions move zero
  bytes and which statistics fields are declared but never incremented.
- **`GpuOptimizer::to_gpu` / `to_cpu`** renamed to `move_to_gpu` / `move_to_cpu`
  (`optirs-gpu`, internal step engine) to match Rust self-convention.
- **SciRS2 dependencies raised from 0.4.0 to 0.6.5** (`scirs2-core`, `scirs2-optimize`,
  `scirs2-neural`, `scirs2-stats`, and the optional `scirs2-metrics` /
  `scirs2-datasets`).
- **Reproducible tests.** Learning-outcome tests that previously depended on entropy
  seeding (and could therefore fail as a lottery over initializations) are pinned to
  fixed seeds, with a determinism regression test.

### Fixed

- **Secure aggregation was numerically wrong as well as insecure**: masks were added and
  never removed, so the "aggregate" was the mean of the client updates plus a pile of
  random noise. Masks now cancel exactly.
- **DARTS progressive discretization selected the wrong operation.** The default path
  took the argmax of *squared* signed logits, which selects the most negative logit.
  Replaced with a plain argmax, with regression tests.
- **`LowLatencyOptimizer::exact_update` zeroed parameters** on every call by treating a
  fresh zero array as the current parameters.
- **A prequential drift baseline absorbed the drift it existed to detect** — its variance
  estimator was an EWMA of the squared deviation from the slow mean, so a single
  k-sigma observation multiplied it by roughly `α·k²` (measured: `1.0e-5` → `70` in one
  observation), masking the shift within about twenty samples. Now Winsorised.
- **Second-order finite-difference steps were not precision-aware.** Second-order stencils
  used the raw configured `1e-5` epsilon; dividing by `h² = 1e-10` puts the roundoff floor
  at about `2e-6` in `f64` and `1.2e3` in `f32` — the `f32` Hessian was pure noise. Fixed
  to a machine-precision-aware step at the stencil's total order.
- **`truncated_newton_direction` materialised the entire Hessian inside every CG
  iteration** (`4n²` objective calls per Hessian-vector product where `4n` suffice).
- **`kfac_hessian_approximation` indexed past the end** of the activation/gradient slices
  instead of returning an error.
- **`jacobian_forward_mode` used a one-sided step** and re-evaluated the objective once per
  input column, inconsistently with the reverse-mode path.
- **K-FAC convolution statistics** were computed from a modulo placeholder instead of a
  real averaged outer product.
- **NAS architecture encoding was lossy**: operation indices are now an injective 0..=40
  mapping with lossless metadata and an explicit decline-to-guess fallback, covered by a
  bijection test.
- **Resource-budget checks compared system-wide RAM against a per-process allocation
  budget**, reporting 671 % memory utilisation and a spurious violation on a default
  configuration.
- **The cross-platform benchmark harness could report infinite throughput**
  (`optirs-core/src/benchmarking/cross_platform_tester.rs`): per-iteration timing
  quantised to zero on fast closures, so the derived rate was a division by zero. Timing
  is now batched across iterations, and an unresolvable duration is an explicit error
  rather than a fabricated rate.
- **Docker benchmark containers exited immediately** because `docker create` with no
  command used the image's default `CMD`.
- **`ResourceConstraints::default` latent bug** and an inverted task-priority sort
  (priorities are now sorted descending) were corrected.
- **`optirs-wasm` WebGPU feature detection** silently returned `false` regardless of real
  browser support, and `WasmGpuOptimizer` had no constructor at all, making its accessors
  unreachable.
- Numerous unguarded index and shape operations now return typed errors instead of
  panicking.

### Removed

> **API removals.** This release deletes a substantial amount of scaffolding that was
> publicly exported but never implemented — types whose methods returned `Ok(())`,
> configuration structs read by nobody, and duplicate definitions of the same concept.
> Code that named these types will no longer compile. Nothing that had a working
> implementation was removed, and no working type was renamed.

- **`optirs-core::privacy`**: the duplicate `SecureAggregator<T>` in
  `federated_privacy/components.rs` (its state was the insecure server-holds-every-mask
  design); `AdvancedFederatedConfig` and roughly 14 of its sub-configuration structs;
  roughly 20 residual duplicate public types; 11 constructor-only shells in
  `components.rs` and 8 dead shells in the private-HPO type module; and 35 generated
  `*_traits.rs` shell files across `enhanced_audit/` and
  `private_hyperparameter_optimization/`, collapsed into one `trait_impls.rs` each.
- **`optirs-core::plugin`**: the `template_generator` and `validator` modules.
- **`optirs-learned`**: roughly 40 dead scaffolding types, and about 13 million dead
  duplicate parameters in the transformer optimizer (which is what made its previously
  `#[ignore]`d creation test fast enough to enable).
- **`optirs-nas`**: roughly 45 dead scaffolding types, including the whole of
  `multi_objective/algorithms.rs` (`IBEA`, `QualityIndicator`, `SmsEmoa`),
  `multi_objective/preference.rs` (`ConstraintHandler` and friends) and 22 types from
  `evaluation/predictor.rs`. All three pre-existing `#[allow(unused_imports)]`
  attributes were removed; the crate now contains no `#[allow]` of any kind.
- **`optirs-tpu`**: the duplicate `xla_compilation` module (folded into `xla/`); the
  `PowerProfiler` scaffolding in the XLA profiling-integration module; the descriptor
  stand-in for a compiled program binary (`encode_program_binary`, `tpu_version_code`,
  `optimization_level_code`, `fnv1a_64`) and the identity `evaluate_reference`, both
  superseded by the real code generator and reference executor; and the
  optimization-level lookup tables `estimate_compute_utilization` /
  `estimate_bandwidth_utilization`, superseded by measured figures.
- **`optirs-gpu`**: `src/kernels.rs` and the SGD/Adam kernel string templates; the
  `thread::sleep` latency simulations in all four vendor memory backends.
- **`optirs-bench`**: the superseded `memory_profiler_integration` and
  `regression_tester_refactored` modules (`regression_tester` remains).
- **Dependencies**: `tokenizers` and `autograd` (both pulled in C/C++ code), and the
  unused `scirs2-linalg`, `scirs2-signal` and `scirs2-series` dependencies.

### Verification

At the close of this cycle, on `--all-features`:

- `cargo check --workspace --all-targets` — 0 warnings
- `cargo clippy --workspace --all-targets` — 0 warnings
- more than 4,200 unit/integration tests passing, plus the doc tests
- `cargo deny check bans` — ok
- no source file at or above 2,000 lines

## [0.3.1] - 2026-03-27

### Changed

- **Version Synchronization** - All lib.rs doc comments now reflect accurate version 0.3.1
- **Documentation** - Updated version references throughout codebase

### Added

- **optirs-wasm README** - Added comprehensive README.md for WebAssembly bindings crate

### Fixed

- **Doc Comment Versions** - Fixed stale 0.1.0 version strings in all crate documentation

---

## [0.3.0] - 2026-03-17

### Added

#### New Optimizers (optirs-core)
- **ReptileOptimizer** - Meta-learning optimizer with inner loop SGD and parameter interpolation
- **MetaSGD** - Per-parameter learnable learning rates with meta-gradient updates

#### New Regularizers (optirs-core)
- **GroupLasso** - Group-wise L2 norm penalty for structured sparsity
- **StructuredSparsity** - Column, row, and block sparsity patterns

#### Optimizer Composition (optirs-core)
- **WeightedOptimizer** - Weighted average of multiple optimizer outputs

#### Continual Learning (optirs-learned)
- **ElasticWeightConsolidation** - Diagonal Fisher information with parameter anchoring (standard + online EWC)
- **ProgressiveNetworks** - Column networks per task with lateral connections

#### Domain-Specific Optimizers (optirs-learned)
- **CVOptimizer** - Spatial-aware LR, channel normalization, progressive resolution
- **NLPOptimizer** - Layer-wise LR decay, warmup/cosine schedule, gradient accumulation
- **AttentionOptimizer** - Head-wise gradient scaling, attention entropy regularization

#### Meta-Learning (optirs-learned)
- **ReptileLearner** - First-order meta-learning with task averaging
- **MetaSGDLearner** - Meta-learning with per-parameter learnable learning rates
- Fixed MAML `compute_gradients` - Now uses finite-difference approximation instead of returning zeros

#### Neural Architecture Search (optirs-nas)
- **MemoryEfficientDARTS** - Partial channel connections and edge normalization (PC-DARTS)
- **RobustDARTS** - Perturbation regularization, collapse detection, early stopping
- **DARTSConfig** - Builder pattern for DARTS variants with temperature scheduling
- **ProgressiveNAS** - Phase-based architecture search with increasing complexity budgets
- **DomainNASEngine** - Domain-specific NAS with pre-configured search spaces for CV, NLP, TimeSeries
- **ArchitectureEmbedder** - Architecture embedding into vector space with cosine similarity search

#### Distributed Training (optirs-core) — Wave 2
- **FedProxOptimizer** - Federated proximal optimizer with configurable mu coefficient (mu=0 degenerates to FedAvg)

#### Learning Rate Schedulers (optirs-core) — Wave 2
- **ViTLayerDecay** - Vision Transformer layer-wise exponential LR decay with warmup + cosine schedule
- **AttentionAwareScheduler** - Transformer component-specific LR scaling (Attention, FeedForward, Embedding, LayerNorm, Output)

#### Gradient & Loss Analysis (optirs-core) — Wave 2
- **GradientFlowAnalyzer** - Gradient flow recording, vanishing/exploding detection, health reports, SVG visualization
- **LossLandscapeAnalyzer** - 2D loss landscape perturbation analysis, sharpness computation, saddle point detection, SVG contour plots

#### Few-Shot Learning (optirs-learned) — Wave 2
- **PrototypicalNetwork** - Encoding, prototype computation, nearest-prototype classification
- **FastAdaptationEngine** - Multi-step gradient adaptation with strategy selection
- **TaskSimilarityCalculator** - Cosine similarity for task representations
- **EpisodicMemoryBank** - Episode storage/retrieval with eviction policies (Performance, LRU, LFU, Age)
- **SupportSetManager** - Diverse support set selection, noise augmentation, quality evaluation

#### Online Meta-Learning (optirs-learned) — Wave 2
- **OnlineMAML** - Continuous task stream meta-learning with staleness decay and buffer management
- **CrossDomainTransfer** - Cross-domain knowledge transfer with shared representations and transferability matrix

### Refactored

- **search_strategies.rs** (2052 lines) split into modular directory structure:
  - `search_strategies/mod.rs` - Trait definitions and re-exports
  - `search_strategies/random.rs` - RandomSearch
  - `search_strategies/evolutionary.rs` - EvolutionarySearch
  - `search_strategies/rl_search.rs` - ReinforcementLearningSearch
  - `search_strategies/differentiable.rs` - DifferentiableSearch (DARTS) + variants
  - `search_strategies/bayesian.rs` - BayesianOptimization
  - `search_strategies/neural_predictor.rs` - NeuralPredictorSearch
  - `search_strategies/progressive.rs` - ProgressiveNAS

### Changed

#### Workspace & Dependencies
- **Bumped workspace version to 0.3.0** - All crates now at version 0.3.0
- Updated all internal crate dependency versions to 0.3.0
- Updated publish script to version 0.3.0

#### Documentation
- Updated TODO.md version references to 0.3.0
- Updated CHANGELOG with 0.3.0 release notes

---

## [0.2.0] - 2026-02-16

### Changed

#### Licensing
- **License changed to Apache-2.0 only** - OptiRS now uses Apache-2.0 as its sole license, aligning with the broader SciRS2 ecosystem
- Consolidated license files into single `LICENSE` file
- Updated all crate manifests and documentation to reflect Apache-2.0 license

#### SciRS2 Integration
- **Updated to SciRS2 v0.3.0** - Upgraded all SciRS2 dependencies from v0.2.0 to v0.3.0
  - `scirs2-core` 0.3.0 - Core scientific computing primitives
  - `scirs2-optimize` 0.3.0 - Base optimization interfaces
  - `scirs2-neural` 0.3.0 - Neural network support
  - `scirs2-metrics` 0.3.0 - Performance monitoring
  - `scirs2-stats` 0.3.0 - Statistical analysis
  - `scirs2-series` 0.3.0 - Time series support
  - `scirs2-datasets` 0.3.0 - Dataset utilities
  - `scirs2-linalg` 0.3.0 - Linear algebra operations
  - `scirs2-signal` 0.3.0 - Signal processing

#### Documentation
- Moved `CLAUDE.md` from repository to temporary directory (per project policy)
- Updated version references across all documentation to v0.3.0
- Updated `MIGRATION_FROM_SCIRS2.md` to reflect new version
- Updated all subcrate documentation headers

#### Build System
- Updated workspace version to 0.2.0 across all crates
- Refined workspace dependency management for SciRS2 v0.3.0
- Updated `optirs-gpu` dependency configuration

### Removed
- Old dual-license files (replaced by single `LICENSE`)
- `LICENSE-APACHE` file (consolidated into `LICENSE`)
- `CLAUDE.md` from repository (moved to temporary directory per policy)

### Maintenance
- Minor version updates to library headers
- Improved consistency across crate versions
- Streamlined license file structure

### Migration Notes
For users upgrading from 0.1.0 to 0.2.0:
- **License Change**: OptiRS is now Apache-2.0 only. Please review your legal requirements if needed
- **SciRS2 Dependencies**: Update all SciRS2 dependencies to v0.3.0
- **No API Changes**: The public API remains fully compatible with v0.1.0
- **No Breaking Changes**: This release is backward compatible at the API level

### Dependencies
All SciRS2 dependencies updated to v0.3.0:
```toml
scirs2-core = "0.3.0"
scirs2-optimize = "0.3.0"
scirs2-neural = "0.3.0"
scirs2-metrics = "0.3.0"
scirs2-stats = "0.3.0"
scirs2-series = "0.3.0"
scirs2-datasets = "0.3.0"
scirs2-linalg = "0.3.0"
scirs2-signal = "0.3.0"
```

---

## [0.1.0] - 2025-12-30

### 🎉 Initial Release

OptiRS v0.1.0 is the first release of a comprehensive ML optimization library for Rust, built exclusively on the SciRS2 scientific computing ecosystem. This release provides 19 production-ready optimizers, comprehensive documentation, full SciRS2 integration, and zero clippy warnings.

### Added

#### Core Optimizers (19 total)

**First-Order Optimizers (17)**
- **SGD** - Stochastic Gradient Descent with momentum and Nesterov acceleration
- **SimdSGD** - SIMD-accelerated SGD (2-4x faster for large arrays)
- **Adam** - Adaptive Moment Estimation
- **AdamW** - Adam with decoupled weight decay
- **AdaDelta** - Adaptive learning rate without manual tuning
- **AdaBound** - Dynamic bounds smoothly transitioning from Adam to SGD
- **RMSprop** - Root Mean Square Propagation
- **Adagrad** - Adaptive Gradient Algorithm
- **LAMB** - Layer-wise Adaptive Moments for batch training
- **LARS** - Layer-wise Adaptive Rate Scaling
- **Lion** - Evolved Sign Momentum optimizer
- **Lookahead** - k steps forward, 1 step back wrapper
- **RAdam** - Rectified Adam with variance rectification
- **Ranger** - RAdam + Lookahead combination
- **SAM** - Sharpness-Aware Minimization
- **SparseAdam** - Adam optimized for sparse gradients
- **GroupedAdam** - Adam with parameter groups

**Second-Order Optimizers (2)**
- **L-BFGS** - Limited-memory Broyden-Fletcher-Goldfarb-Shanno
- **Newton-CG** - Newton Conjugate Gradient with trust region

#### Learning Rate Schedulers
- **ExponentialDecay** - Exponential learning rate decay
- **StepDecay** - Step-wise reduction
- **CosineAnnealing** - Cosine annealing schedule
- **LinearWarmup** - Linear warmup with decay
- **OneCycle** - One cycle learning rate policy

#### Performance Features
- **SIMD Acceleration** - 2-4x speedup for large parameter arrays
  - Automatic SIMD vectorization for f32/f64
  - Threshold-based activation (16 elements for f32, 8 for f64)

- **Parallel Processing** - 4-8x speedup for parameter groups
  - Multi-core parameter group processing
  - Automatic work distribution across CPU cores

- **Memory-Efficient Operations**
  - Gradient accumulation for micro-batch training
  - Chunked parameter processing for billion-parameter models
  - Memory usage estimation and recommendations

- **GPU Framework** - Multi-backend support foundation
  - CUDA, Metal, OpenCL, WebGPU backends
  - GPU context management and initialization
  - Tensor cores and mixed-precision support

#### Production Tools
- **Metrics & Monitoring**
  - Real-time optimizer performance tracking
  - Gradient statistics (mean, std dev, norm, sparsity)
  - Parameter statistics (update magnitude, relative change)
  - Convergence detection with moving averages
  - Export to JSON and CSV formats

- **Comprehensive Benchmarking**
  - Statistical performance analysis with Criterion.rs
  - Memory profiling and leak detection
  - Cross-platform performance testing
  - Regression detection

#### Module Organization
- `optirs-core` - Core optimization algorithms (Production Ready)
- `optirs-bench` - Benchmarking and performance analysis (Production Ready)
- `optirs-gpu` - GPU acceleration framework (In Development)
- `optirs-tpu` - TPU coordination (Framework Ready)
- `optirs-learned` - Learned optimizers and meta-learning (Research Phase)
- `optirs-nas` - Neural Architecture Search (Research Phase)

### Features

#### SciRS2 Ecosystem Integration
Complete integration with SciRS2 v0.3.0:
- ✅ Arrays: `scirs2_core::ndarray` exclusively (NO direct ndarray)
- ✅ Random: `scirs2_core::random` exclusively (NO direct rand)
- ✅ Numerical: `scirs2_core::numeric` for all numerical traits
- ✅ SIMD: `scirs2_core::simd_ops` for vectorization
- ✅ Parallel: `scirs2_core::parallel_ops` for multi-core
- ✅ GPU: `scirs2_core::gpu` abstractions
- ✅ Metrics: `scirs2_core::metrics` for monitoring

#### Quality Assurance
- **1,134 tests passing** (100% pass rate)
  - 1,061 unit tests
  - 73 doc tests
- **Zero clippy warnings** - Production-ready code quality
- **100% public API documentation**
- **Comprehensive examples** - All features demonstrated

### Performance

#### Benchmarks
- **SGD**: < 10ns per parameter update
- **Adam**: < 50ns per parameter update
- **SIMD variants**: 2-4x faster on large arrays
- **Parallel processing**: 4-8x speedup on multi-core

#### Memory Efficiency
- Optimizer state: < 2x parameter memory
- Zero-copy operations where possible
- Efficient gradient accumulation
- Chunked processing for large models

### Documentation

#### Comprehensive Documentation
- **README.md** - Project overview and quick start
- **USAGE_GUIDE.md** - Comprehensive usage guide (8000+ words)
- **MIGRATION_FROM_SCIRS2.md** - Migration guide for SciRS2 users
- **SCIRS2_INTEGRATION_POLICY.md** - Critical dependency policy
- **API Documentation** - 100% coverage with examples
- **Examples** - 4 comprehensive example files

### Dependencies

#### SciRS2 Ecosystem (v0.3.0)
- `scirs2-core` - Core scientific primitives (REQUIRED)
- `scirs2-optimize` - Base optimization interfaces (REQUIRED)
- `scirs2-neural` - Neural network support
- `scirs2-metrics` - Performance monitoring
- `scirs2-stats` - Statistical analysis
- `scirs2-series` - Time series support
- `scirs2-datasets` - Dataset utilities

#### External Dependencies
- Serialization: `serde`, `serde_json`, `oxicode`
- Testing: `approx`, `criterion`, `tokio-test`
- GPU: `cudarc`, `opencl3`, `wgpu`
- ML: `tokenizers`, `autograd`
- Utilities: `thiserror`, `anyhow`, `chrono`

### Policy Compliance

✅ **COOLJAPAN Policy**
- Using `oxiblas` via scirs2-core (NO openblas)

✅ **Latest Crates Policy**
- All dependencies at latest compatible versions

✅ **Workspace Policy**
- Version managed via `workspace = true`

✅ **No Warnings Policy**
- Zero clippy warnings
- Clean compilation with all features

### Platform Support

| Platform | Core | GPU | TPU | Learned | NAS | Bench |
|----------|------|-----|-----|---------|-----|-------|
| Linux    | ✅   | 🚧  | 🚧  | 🔬      | 🔬  | ✅    |
| macOS    | ✅   | 🚧  | ❌  | 🔬      | 🔬  | ✅    |
| Windows  | ✅   | 🚧  | ❌  | 🔬      | 🔬  | ✅    |

Legend: ✅ Production Ready | 🚧 In Development | 🔬 Research Phase | ❌ Not Supported

### Contributors

- COOLJAPAN OU (Team Kitasan)
- SciRS2 Integration Team
- Rust ML Community

### Acknowledgments

- SciRS2 project for providing robust scientific computing foundation
- Rust ML community for feedback and support
- Open source optimization algorithm researchers

---

[0.3.2]: https://github.com/cool-japan/optirs/releases/tag/v0.3.2
[0.3.1]: https://github.com/cool-japan/optirs/releases/tag/v0.3.1
[0.3.0]: https://github.com/cool-japan/optirs/releases/tag/v0.3.0
[0.2.0]: https://github.com/cool-japan/optirs/releases/tag/v0.2.0
[0.1.0]: https://github.com/cool-japan/optirs/releases/tag/v0.1.0
