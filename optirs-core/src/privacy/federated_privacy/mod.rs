// Federated Privacy Algorithms
//
// This module implements privacy-preserving algorithms specifically designed
// for federated learning scenarios, including secure aggregation, client-side
// differential privacy, privacy amplification through federation, advanced
// threat modeling, and cross-silo federated learning with heterogeneous clients.

pub mod adaptation;
pub mod components;
pub mod composition;
pub mod config;
pub mod coordinator;
pub mod validation;

// Re-export main types and structs for public API
pub use adaptation::DEFAULT_INNER_LEARNING_RATE;
pub use components::*;
pub use composition::ComposedPrivacyCost;
pub use config::*;
pub use coordinator::*;
pub use validation::{validate_composition_method, validate_sampling_strategy};

// Re-export the main coordinator for backwards compatibility
pub use coordinator::FederatedPrivacyCoordinator;
