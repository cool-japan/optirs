// Adaptive Neural Architecture Search System
//
// This module implements an advanced NAS system that continuously learns from
// optimization performance to automatically design better optimizer architectures.
//
// The system is organized into focused modules:
//
// - `config`: Configuration types and settings (✓ v1.0.0)
// - `searcher`: Performance-aware architecture search components (ROADMAP v1.1.0+)
// - `generator`: Architecture generation and candidate creation (ROADMAP v1.1.0+)
// - `database`: Performance database and storage (ROADMAP v1.1.0+)
// - `optimizer`: Multi-objective optimization (ROADMAP v1.1.0+)
// - `predictor`: Performance prediction ensemble (ROADMAP v1.1.0+)
// - `adaptation`: Continuous learning and adaptation (ROADMAP v1.1.0+)
// - `quality`: Architecture quality assessment (ROADMAP v1.1.0+)
// - `space`: Search space management (ROADMAP v1.1.0+)
// - `state`: System state tracking (ROADMAP v1.1.0+)

pub mod config;

// ROADMAP (v1.1.0+): Additional modules planned for future releases from adaptive_nas_system.rs
// pub mod searcher;     // PerformanceAwareSearcher, SearchHistory, etc.
// pub mod generator;    // LearningBasedGenerator, ArchitectureCandidateGenerator, etc.
// pub mod database;     // ArchitecturePerformanceDatabase, etc.
// pub mod optimizer;    // MultiObjectiveArchitectureOptimizer, etc.
// pub mod predictor;    // PerformancePredictorEnsemble, PerformanceModel, etc.
// pub mod adaptation;   // ContinuousAdaptationEngine, AdaptationStrategy, etc.
// pub mod quality;      // ArchitectureQualityAssessor, etc.
// pub mod space;        // DynamicSearchSpaceManager, etc.
// pub mod state;        // NASSystemStateTracker, NASSystemState, etc.

// Re-export configuration types
pub use config::{AdaptiveNASConfig, BudgetAllocationStrategy, QualityCriterion};

// ROADMAP (v1.1.0+): Additional re-exports once modules are created
// pub use searcher::*;
// pub use generator::*;
// pub use database::*;
// pub use optimizer::*;
// pub use predictor::*;
// pub use adaptation::*;
// pub use quality::*;
// pub use space::*;
// pub use state::*;