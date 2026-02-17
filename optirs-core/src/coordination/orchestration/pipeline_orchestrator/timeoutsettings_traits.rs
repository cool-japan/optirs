//! # TimeoutSettings - Trait Implementations
//!
//! This module contains trait implementations for `TimeoutSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

use super::types::TimeoutSettings;

impl Default for TimeoutSettings {
    fn default() -> Self {
        use std::time::Duration;
        Self {
            global_timeout: Some(Duration::from_secs(3600)),
            stage_timeouts: HashMap::new(),
            heartbeat_timeout: Duration::from_secs(30),
            response_timeout: Duration::from_secs(60),
        }
    }
}
