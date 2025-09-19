// Filter Rules, Conditions, and Actions
//
// This module provides filter rule definitions, condition evaluation,
// and action execution for event filtering systems.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime};

/// Filter rules configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterRulesConfig {
    /// Available filter rules
    pub rules: Vec<FilterRule>,
    /// Rule execution order
    pub execution_order: RuleExecutionOrder,
    /// Rule priority management
    pub priority_management: RulePriorityManagement,
    /// Rule conflict resolution
    pub conflict_resolution: RuleConflictResolution,
    /// Rule validation settings
    pub validation: RuleValidation,
}

impl Default for FilterRulesConfig {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            execution_order: RuleExecutionOrder::Priority,
            priority_management: RulePriorityManagement::default(),
            conflict_resolution: RuleConflictResolution::default(),
            validation: RuleValidation::default(),
        }
    }
}

/// Individual filter rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterRule {
    /// Rule identifier
    pub rule_id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Filter condition
    pub condition: FilterCondition,
    /// Filter action
    pub action: FilterAction,
    /// Rule priority
    pub priority: i32,
    /// Rule status
    pub status: RuleStatus,
    /// Rule metadata
    pub metadata: RuleMetadata,
    /// Performance metrics
    pub performance: RulePerformanceMetrics,
}

/// Filter condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterCondition {
    /// Simple field comparison
    FieldComparison {
        field: String,
        operator: ComparisonOperator,
        value: FilterValue,
    },
    /// Pattern matching
    PatternMatch {
        field: String,
        pattern: String,
        flags: PatternFlags,
    },
    /// Range condition
    Range {
        field: String,
        min: FilterValue,
        max: FilterValue,
        inclusive: bool,
    },
    /// Set membership
    SetMembership {
        field: String,
        values: HashSet<FilterValue>,
        negate: bool,
    },
    /// Composite condition
    Composite {
        operator: LogicalOperator,
        conditions: Vec<FilterCondition>,
    },
    /// Custom expression
    Expression {
        expression: String,
        variables: HashMap<String, FilterValue>,
    },
    /// Time-based condition
    TimeBased {
        field: String,
        time_range: TimeRange,
        timezone: Option<String>,
    },
    /// Statistical condition
    Statistical {
        field: String,
        statistic: StatisticType,
        threshold: f64,
        window: Duration,
    },
}

/// Comparison operators for field comparisons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Contains,
    StartsWith,
    EndsWith,
    Matches,
}

/// Logical operators for composite conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalOperator {
    And,
    Or,
    Not,
    Xor,
}

/// Filter value types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FilterValue {
    String(String),
    Integer(i64),
    Float(String), // Stored as string to avoid float comparison issues
    Boolean(bool),
    Null,
    Array(Vec<FilterValue>),
    Object(HashMap<String, FilterValue>),
}

/// Pattern matching flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternFlags {
    /// Case insensitive matching
    pub case_insensitive: bool,
    /// Multiline mode
    pub multiline: bool,
    /// Dot matches newline
    pub dot_all: bool,
    /// Extended syntax
    pub extended: bool,
}

impl Default for PatternFlags {
    fn default() -> Self {
        Self {
            case_insensitive: false,
            multiline: false,
            dot_all: false,
            extended: false,
        }
    }
}

/// Time range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time
    pub start: SystemTime,
    /// End time
    pub end: SystemTime,
    /// Relative to current time
    pub relative: bool,
}

/// Statistical types for statistical conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticType {
    Count,
    Sum,
    Average,
    Minimum,
    Maximum,
    StandardDeviation,
    Percentile(f64),
    Rate,
    Frequency,
}

/// Filter actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterAction {
    /// Accept the event
    Accept,
    /// Reject the event
    Reject,
    /// Transform the event
    Transform {
        transformations: Vec<EventTransformation>,
    },
    /// Route to specific destination
    Route {
        destination: String,
        priority: Option<i32>,
    },
    /// Delay the event
    Delay { duration: Duration, reason: String },
    /// Split into multiple events
    Split { split_rules: Vec<SplitRule> },
    /// Aggregate with other events
    Aggregate {
        aggregation_key: String,
        aggregation_window: Duration,
    },
}

/// Event transformation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventTransformation {
    /// Add field
    AddField { field: String, value: FilterValue },
    /// Remove field
    RemoveField { field: String },
    /// Rename field
    RenameField { old_name: String, new_name: String },
    /// Transform field value
    TransformField {
        field: String,
        transformation: ValueTransformation,
    },
    /// Format field
    FormatField { field: String, format: String },
    /// Math operation
    MathOperation {
        target_field: String,
        operation: MathOperationType,
        operands: Vec<String>,
    },
}

/// Value transformation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueTransformation {
    /// Convert to uppercase
    ToUpperCase,
    /// Convert to lowercase
    ToLowerCase,
    /// Trim whitespace
    Trim,
    /// Parse as number
    ParseNumber,
    /// Parse as date
    ParseDate { format: String },
    /// Substring extraction
    Substring { start: usize, length: Option<usize> },
    /// Regular expression replacement
    RegexReplace {
        pattern: String,
        replacement: String,
    },
}

/// Mathematical operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MathOperationType {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
    Round,
    Floor,
    Ceiling,
}

/// Split rule for event splitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitRule {
    /// Split strategy
    pub strategy: SplitStrategy,
    /// Target field for splitting
    pub field: String,
    /// Split parameters
    pub parameters: HashMap<String, FilterValue>,
}

/// Event splitting strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplitStrategy {
    /// Split by delimiter
    ByDelimiter {
        delimiter: String,
        max_splits: Option<usize>,
    },
    /// Split by fixed size
    ByFixedSize { size: usize, overlap: usize },
    /// Split by pattern
    ByPattern {
        pattern: String,
        include_separator: bool,
    },
    /// Split by array elements
    ByArrayElements,
    /// Custom splitting logic
    Custom { logic: String },
}

/// Rule execution order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleExecutionOrder {
    /// Execute by priority (highest first)
    Priority,
    /// Execute by creation time
    CreationTime,
    /// Execute by last modified time
    LastModified,
    /// Execute in specified order
    Explicit(Vec<String>),
    /// Execute by rule dependencies
    Dependency,
}

/// Rule priority management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulePriorityManagement {
    /// Enable dynamic priority adjustment
    pub dynamic_adjustment: bool,
    /// Priority adjustment strategy
    pub adjustment_strategy: PriorityAdjustmentStrategy,
    /// Priority conflict resolution
    pub conflict_resolution: PriorityConflictResolution,
    /// Priority bounds
    pub min_priority: i32,
    pub max_priority: i32,
}

impl Default for RulePriorityManagement {
    fn default() -> Self {
        Self {
            dynamic_adjustment: false,
            adjustment_strategy: PriorityAdjustmentStrategy::PerformanceBased,
            conflict_resolution: PriorityConflictResolution::FirstWins,
            min_priority: -1000,
            max_priority: 1000,
        }
    }
}

/// Priority adjustment strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorityAdjustmentStrategy {
    /// Adjust based on performance metrics
    PerformanceBased,
    /// Adjust based on usage frequency
    UsageFrequency,
    /// Adjust based on success rate
    SuccessRate,
    /// Adjust based on resource consumption
    ResourceConsumption,
    /// Custom adjustment logic
    Custom(String),
}

/// Priority conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorityConflictResolution {
    /// First rule wins
    FirstWins,
    /// Last rule wins
    LastWins,
    /// Merge rule results
    Merge,
    /// Execute all rules
    ExecuteAll,
    /// Highest priority wins
    HighestPriority,
}

/// Rule conflict resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleConflictResolution {
    /// Conflict detection enabled
    pub detection_enabled: bool,
    /// Resolution strategy
    pub strategy: ConflictResolutionStrategy,
    /// Conflict notification
    pub notification: ConflictNotification,
    /// Conflict logging
    pub logging: bool,
}

impl Default for RuleConflictResolution {
    fn default() -> Self {
        Self {
            detection_enabled: true,
            strategy: ConflictResolutionStrategy::FirstMatch,
            notification: ConflictNotification::default(),
            logging: true,
        }
    }
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    /// Use first matching rule
    FirstMatch,
    /// Use last matching rule
    LastMatch,
    /// Use highest priority rule
    HighestPriority,
    /// Combine rule results
    Combine,
    /// Execute all conflicting rules
    ExecuteAll,
    /// Reject on conflict
    Reject,
}

/// Conflict notification settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictNotification {
    /// Enable notifications
    pub enabled: bool,
    /// Notification threshold
    pub threshold: usize,
    /// Notification channels
    pub channels: Vec<String>,
}

impl Default for ConflictNotification {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 1,
            channels: Vec::new(),
        }
    }
}

/// Rule validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleValidation {
    /// Enable syntax validation
    pub syntax_validation: bool,
    /// Enable semantic validation
    pub semantic_validation: bool,
    /// Enable performance validation
    pub performance_validation: bool,
    /// Validation strictness
    pub strictness: ValidationStrictness,
    /// Custom validation rules
    pub custom_validators: Vec<String>,
}

impl Default for RuleValidation {
    fn default() -> Self {
        Self {
            syntax_validation: true,
            semantic_validation: true,
            performance_validation: false,
            strictness: ValidationStrictness::Medium,
            custom_validators: Vec::new(),
        }
    }
}

/// Validation strictness levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStrictness {
    Low,
    Medium,
    High,
    Strict,
}

/// Rule status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleStatus {
    Active,
    Inactive,
    Draft,
    Deprecated,
    Testing,
}

/// Rule metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleMetadata {
    /// Creator information
    pub creator: String,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modified timestamp
    pub modified_at: SystemTime,
    /// Version information
    pub version: String,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Custom attributes
    pub attributes: HashMap<String, String>,
}

/// Rule performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulePerformanceMetrics {
    /// Execution count
    pub execution_count: u64,
    /// Match count
    pub match_count: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Error count
    pub error_count: u64,
    /// Last execution timestamp
    pub last_execution: Option<SystemTime>,
}

impl Default for RulePerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_count: 0,
            match_count: 0,
            total_execution_time: Duration::from_secs(0),
            average_execution_time: Duration::from_secs(0),
            error_count: 0,
            last_execution: None,
        }
    }
}
