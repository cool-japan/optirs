// Layout Management Module for TPU Pod Topology
//
// This module provides comprehensive layout management functionality for TPU pod coordination,
// including physical layout, thermal management, power distribution, logical topology,
// and layout algorithms.

pub mod physical;

// Re-export main types explicitly to avoid ambiguous glob re-exports
// Users can access submodules directly via layout::physical::management, layout::thermal::management, etc.
