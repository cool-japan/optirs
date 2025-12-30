// Event Ordering Module
//
// This module provides comprehensive event ordering functionality for TPU synchronization systems.
// It includes ordering types, enforcement, windows, sequences, gap detection, duplicate detection,
// buffering capabilities, and manager implementations.

pub mod buffering;
pub mod duplicate_detection;
pub mod enforcement;
pub mod gap_detection;
pub mod managers;
pub mod ordering_types;
pub mod sequences;
pub mod windows;

pub use self::buffering::*;
pub use self::duplicate_detection::*;
pub use self::enforcement::*;
pub use self::gap_detection::*;
pub use self::managers::*;
pub use self::ordering_types::*;
pub use self::sequences::*;
pub use self::windows::*;

// Additional type exports for compatibility

#[derive(Debug, Clone, Default)]
pub struct EventOrdering;

#[derive(Debug, Clone, Default)]
pub struct EventOrderingBuilder;

#[derive(Debug, Clone, Default)]
pub struct OrderingBuffering;

#[derive(Debug, Clone, Default)]
pub struct OrderingMetrics;

#[derive(Debug, Clone, Default)]
pub struct OrderingPresets;

#[derive(Debug, Clone, Default)]
pub struct OrderingStrategies;

#[derive(Debug, Clone, Default)]
pub struct SequenceManagement;

#[derive(Debug, Clone, Default)]
pub struct GapDetection;
