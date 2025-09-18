// Routing table management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Routing tables configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTables {
    /// Static routes
    pub static_routes: HashMap<String, StaticRoute>,
    /// Dynamic routes
    pub dynamic_routes: HashMap<String, DynamicRoute>,
    /// Route priorities
    pub priorities: HashMap<String, u32>,
    /// Route metadata
    pub metadata: HashMap<String, RouteMetadata>,
}

impl Default for RoutingTables {
    fn default() -> Self {
        Self {
            static_routes: HashMap::new(),
            dynamic_routes: HashMap::new(),
            priorities: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Static route configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticRoute {
    /// Route path
    pub path: String,
    /// Target endpoint
    pub target: String,
    /// Route priority
    pub priority: u32,
    /// Route metadata
    pub metadata: HashMap<String, String>,
}

/// Dynamic route configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicRoute {
    /// Route pattern
    pub pattern: String,
    /// Target selection strategy
    pub target_selection: TargetSelection,
    /// Route TTL
    pub ttl: Duration,
    /// Route conditions
    pub conditions: Vec<RouteCondition>,
}

/// Target selection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TargetSelection {
    /// First available
    FirstAvailable,
    /// Best performance
    BestPerformance,
    /// Least loaded
    LeastLoaded,
    /// Custom selection
    Custom(String),
}

/// Route condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Condition value
    pub value: String,
    /// Condition operator
    pub operator: ConditionOperator,
}

/// Condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    /// Header condition
    Header(String),
    /// Query parameter condition
    QueryParam(String),
    /// Path condition
    Path,
    /// Method condition
    Method,
    /// Custom condition
    Custom(String),
}

/// Condition operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionOperator {
    /// Equals
    Equals,
    /// Not equals
    NotEquals,
    /// Contains
    Contains,
    /// Starts with
    StartsWith,
    /// Ends with
    EndsWith,
    /// Regex match
    Regex,
}

/// Route metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteMetadata {
    /// Route description
    pub description: String,
    /// Route tags
    pub tags: Vec<String>,
    /// Route owner
    pub owner: String,
    /// Creation time
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified time
    pub modified_at: chrono::DateTime<chrono::Utc>,
}

impl Default for RouteMetadata {
    fn default() -> Self {
        let now = chrono::Utc::now();
        Self {
            description: String::new(),
            tags: Vec::new(),
            owner: "system".to_string(),
            created_at: now,
            modified_at: now,
        }
    }
}