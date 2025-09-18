// Federated Privacy Algorithms
//
// This module implements privacy-preserving algorithms specifically designed
// for federated learning scenarios, including secure aggregation, client-side
// differential privacy, privacy amplification through federation, advanced
// threat modeling, and cross-silo federated learning with heterogeneous clients.

pub mod components;
pub mod config;
pub mod coordinator;

// Re-export main types and structs for public API
pub use components::*;
pub use config::*;
pub use coordinator::*;

// Re-export the main coordinator for backwards compatibility
pub use coordinator::FederatedPrivacyCoordinator;
