//! Data types of the enhanced audit system.
//!
//! The behaviour that used to live here in constructor-only shells now lives in
//! focused sibling modules, each of which documents the fake it replaced:
//!
//! | moved type | new home |
//! |---|---|
//! | `MerkleTree`, `AuditChain`, `AuditTrail` | [`super::integrity`] |
//! | `FormalVerificationEngine`, `TheoremProver` | [`super::verification`] |
//! | `SystemModel`, `ModelChecker` | [`super::model_checking`] |
//! | `ProofSystem`, `CryptographicProofGenerator` | [`super::proofs`] |
//! | `RegulatoryComplianceChecker`, `ComplianceMonitor`, `RegulationChecker` | [`super::compliance`] |
//! | `PrivacyBudgetTracker`, `BudgetForecastingModel`, `PredictionModel` | [`super::budget`] |
//! | `MonitoringDashboard` | [`super::dashboard`] |
//!
//! All of them are still re-exported from `privacy::enhanced_audit`, so paths
//! that went through the module root are unchanged.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use super::budget::PrivacyBudgetTracker;
use super::compliance::{ComplianceMonitor, RegulatoryComplianceChecker};
use super::dashboard::MonitoringDashboard;
use super::functions::{
    AxiomVerifyFn, CryptoProofGenerateFn, CryptoProofVerifyFn, ProofGenerateFn,
    ProofStrategyApplyFn, ProofVerifyFn, TransitionLogicFn, VerifyFn,
};
use super::hashing::{canonical_event_bytes, hmac_sha256, random_key, Digest32};
use super::integrity::AuditTrail;
use super::model_checking::ModelCheckOutcome;
use super::proofs::{CryptographicProofGenerator, SHA256_INTEGRITY};
use super::verification::FormalVerificationEngine;

/// System property for model checking
#[derive(Debug, Clone)]
pub struct SystemProperty {
    /// Property name
    pub name: String,
    /// Formal specification (see [`super::model_checking`] for the grammar)
    pub specification: String,
    /// Property type
    pub property_type: PropertyType,
}
/// Rule severity levels
#[derive(Debug, Clone, Copy)]
pub enum RuleSeverity {
    /// Critical violation requiring immediate action
    Critical,
    /// High severity violation
    High,
    /// Medium severity violation
    Medium,
    /// Low severity violation
    Low,
    /// Informational only
    Info,
}
/// Compliance frameworks supported
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceFramework {
    /// General Data Protection Regulation (EU)
    GDPR,
    /// California Consumer Privacy Act
    CCPA,
    /// Health Insurance Portability and Accountability Act
    HIPAA,
    /// Sarbanes-Oxley Act
    SOX,
    /// Federal Information Security Management Act
    FISMA,
    /// ISO 27001
    ISO27001,
    /// NIST Privacy Framework
    NISTPrivacy,
    /// Custom compliance framework
    Custom(String),
}
/// Individual compliance finding
#[derive(Debug, Clone)]
pub struct ComplianceFinding {
    /// Finding identifier
    pub id: String,
    /// Rule or requirement violated/satisfied
    pub rule: String,
    /// Compliance status
    pub status: ComplianceStatus,
    /// Evidence
    pub evidence: Vec<String>,
    /// Impact level
    pub impact: ImpactLevel,
}
/// Cryptographic keys for proof generation
///
/// Constructors live in [`super::proofs`], including
/// [`CryptographicKeys::generate`] which draws a MAC key from OS entropy.
pub struct CryptographicKeys {
    /// Signing keys
    pub signing_keys: HashMap<String, Vec<u8>>,
    /// Verification keys
    pub verification_keys: HashMap<String, Vec<u8>>,
    /// Encryption keys
    pub encryption_keys: HashMap<String, Vec<u8>>,
}
/// Impact levels for findings
#[derive(Debug, Clone, Copy)]
pub enum ImpactLevel {
    /// Very high impact
    VeryHigh,
    /// High impact
    High,
    /// Medium impact
    Medium,
    /// Low impact
    Low,
    /// Very low impact
    VeryLow,
}
/// Types of prediction models
#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    /// Linear regression (the only family implemented)
    LinearRegression,
    /// ARIMA model (not implemented; selecting it returns an error)
    ARIMA,
    /// Neural network (not implemented; selecting it returns an error)
    NeuralNetwork,
    /// Random forest (not implemented; selecting it returns an error)
    RandomForest,
}
/// Proof strategy for theorem proving
pub struct ProofStrategy<T: Float + Debug + Send + Sync + 'static> {
    /// Strategy name
    pub name: String,
    /// Strategy application function
    pub apply_fn: ProofStrategyApplyFn<T>,
}
/// Compliance assessment result
#[derive(Debug, Clone)]
pub struct ComplianceAssessment {
    /// Framework assessed
    pub framework: ComplianceFramework,
    /// Fraction of applied checks that passed (0.0 - 1.0)
    pub compliance_score: f64,
    /// Detailed findings
    pub findings: Vec<ComplianceFinding>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
}
/// Dashboard configuration
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    /// Refresh interval (seconds)
    pub refresh_interval: u32,
    /// Historical data retention (hours)
    pub history_retention_hours: u32,
    /// Alert thresholds, keyed by metric name
    pub alert_thresholds: HashMap<String, AlertThreshold>,
    /// Dashboard layout
    pub layout: DashboardLayout,
}
/// Verification result
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether verification passed
    pub verified: bool,
    /// Commitment to the values that were checked, when the rule reads them
    pub proof: Option<Vec<u8>>,
    /// Verification message
    pub message: String,
    /// Confidence in the verdict (1.0 for an exact predicate)
    pub confidence: f64,
}
/// Remediation status
#[derive(Debug, Clone)]
pub enum RemediationStatus {
    /// Violation detected but not remediated
    Open,
    /// Remediation in progress
    InProgress,
    /// Violation remediated
    Resolved,
    /// False positive
    FalsePositive,
    /// Accepted risk
    AcceptedRisk,
}
/// Report formats
#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    /// JSON format
    JSON,
    /// XML format
    XML,
    /// PDF format
    PDF,
    /// HTML format
    HTML,
    /// CSV format
    CSV,
}
/// Types of audit events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Privacy budget allocation
    PrivacyBudgetAllocation,
    /// Privacy budget consumption
    PrivacyBudgetConsumption,
    /// Gradient computation
    GradientComputation,
    /// Model parameter update
    ModelParameterUpdate,
    /// Data access
    DataAccess,
    /// User consent
    UserConsent,
    /// Data deletion
    DataDeletion,
    /// Anonymization process
    AnonymizationProcess,
    /// Security incident
    SecurityIncident,
    /// Compliance check
    ComplianceCheck,
    /// Configuration change
    ConfigurationChange,
    /// System startup/shutdown
    SystemLifecycle,
}
/// Audit query criteria
#[derive(Debug, Clone)]
pub struct AuditQueryCriteria {
    /// Filter by actor
    pub actor: Option<String>,
    /// Filter by event type
    pub event_type: Option<AuditEventType>,
    /// Start time filter
    pub start_time: Option<u64>,
    /// End time filter
    pub end_time: Option<u64>,
    /// Substring search in descriptions
    pub text_search: Option<String>,
}
impl AuditQueryCriteria {
    /// Criteria that match every event.
    pub fn any() -> Self {
        Self {
            actor: None,
            event_type: None,
            start_time: None,
            end_time: None,
            text_search: None,
        }
    }
}
impl Default for AuditQueryCriteria {
    fn default() -> Self {
        Self::any()
    }
}
/// Compliance status for events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStatus {
    /// Compliant with framework
    Compliant,
    /// Non-compliant with framework
    NonCompliant(String),
    /// Requires manual review
    RequiresReview(String),
    /// Not applicable
    NotApplicable,
}
/// Verification criticality levels
#[derive(Debug, Clone, Copy)]
pub enum VerificationCriticality {
    /// Must be verified for safety
    Safety,
    /// Must be verified for correctness
    Correctness,
    /// Should be verified for performance
    Performance,
    /// Optional verification
    Optional,
}
/// Cryptographic proof requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofRequirements {
    /// Require zero-knowledge proofs (not implemented; requesting it errors)
    pub zero_knowledge_proofs: bool,
    /// Require non-repudiation proofs (not implemented; requesting it errors)
    pub non_repudiation: bool,
    /// Require integrity proofs (implemented: SHA-256 / HMAC-SHA256)
    pub integrity_proofs: bool,
    /// Require confidentiality proofs (not implemented; requesting it errors)
    pub confidentiality_proofs: bool,
    /// Require completeness proofs (implemented: the commitment covers the
    /// whole released vector, length included)
    pub completeness_proofs: bool,
}
impl Default for ProofRequirements {
    fn default() -> Self {
        Self {
            zero_knowledge_proofs: false,
            non_repudiation: false,
            integrity_proofs: true,
            confidentiality_proofs: false,
            completeness_proofs: true,
        }
    }
}
/// Types of dashboard metrics
#[derive(Debug, Clone, Copy)]
pub enum MetricType {
    /// Counter metric (monotonically increasing)
    Counter,
    /// Gauge metric (can increase or decrease)
    Gauge,
    /// Histogram metric
    Histogram,
    /// Rate metric
    Rate,
}
/// Threshold direction
#[derive(Debug, Clone, Copy)]
pub enum ThresholdDirection {
    /// Alert when value exceeds threshold
    Above,
    /// Alert when value drops below threshold
    Below,
}
/// Alert severity levels
#[derive(Debug, Clone, Copy)]
pub enum AlertSeverity {
    /// Critical alert requiring immediate action
    Critical,
    /// Warning alert
    Warning,
    /// Informational alert
    Info,
}
/// Risk factor
#[derive(Debug, Clone)]
pub struct RiskFactor {
    /// Factor name
    pub name: String,
    /// Factor weight
    pub weight: f64,
    /// Factor value
    pub value: f64,
    /// Factor description
    pub description: String,
}
/// Types of consumption patterns
#[derive(Debug, Clone, Copy)]
pub enum PatternType {
    /// Steady consumption
    Steady,
    /// Bursty consumption
    Bursty,
    /// Periodic consumption
    Periodic,
    /// Irregular consumption
    Irregular,
}
/// Privacy context at time of event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyContext {
    /// Epsilon spent by the operation this event describes
    pub epsilon_budget: f64,
    /// Delta the guarantee is quoted at
    pub delta_budget: f64,
    /// Privacy mechanism used
    pub privacy_mechanism: String,
    /// Data minimization status
    pub data_minimization: bool,
    /// Purpose limitation compliance
    pub purpose_limitation: bool,
    /// Storage limitation compliance
    pub storage_limitation: bool,
}
/// Risk levels for compliance
#[derive(Debug, Clone, Copy)]
pub enum RiskLevel {
    /// Very high risk
    VeryHigh,
    /// High risk
    High,
    /// Medium risk
    Medium,
    /// Low risk
    Low,
    /// Very low risk
    VeryLow,
}
/// Approval levels for remediation
#[derive(Debug, Clone, Copy)]
pub enum ApprovalLevel {
    /// Automatic execution
    Automatic,
    /// Requires operator approval
    Operator,
    /// Requires supervisor approval
    Supervisor,
    /// Requires executive approval
    Executive,
}
/// Budget alert
#[derive(Debug, Clone)]
pub struct BudgetAlert {
    /// Alert identifier
    pub id: String,
    /// Alert type
    pub alert_type: BudgetAlertType,
    /// Alert message
    pub message: String,
    /// Severity level
    pub severity: AlertSeverity,
    /// Timestamp
    pub timestamp: u64,
    /// Acknowledged
    pub acknowledged: bool,
}
/// Alert threshold configuration
#[derive(Debug, Clone)]
pub struct AlertThreshold {
    /// Metric name
    pub metric: String,
    /// Warning threshold
    pub warning: f64,
    /// Critical threshold
    pub critical: f64,
    /// Threshold direction
    pub direction: ThresholdDirection,
}
/// Formal verification rule
pub struct FormalVerificationRule<T: Float + Debug + Send + Sync + 'static> {
    /// Rule name
    pub name: String,
    /// Formal specification
    pub specification: String,
    /// Verification function
    pub verify_fn: VerifyFn<T>,
    /// Rule criticality
    pub criticality: VerificationCriticality,
}
/// Risk assessment
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    /// Overall risk score (0.0 - 1.0)
    pub risk_score: f64,
    /// Risk factors
    pub risk_factors: Vec<RiskFactor>,
    /// Mitigation recommendations
    pub mitigations: Vec<String>,
    /// Residual risk after the listed mitigations have been executed
    pub residual_risk: f64,
}
/// Budget allocation for specific purpose
#[derive(Debug, Clone)]
pub struct BudgetAllocation {
    /// Purpose identifier
    pub purpose: String,
    /// Allocated epsilon
    pub allocated_epsilon: f64,
    /// Delta the allocation is quoted at
    pub allocated_delta: f64,
    /// Consumed epsilon
    pub consumed_epsilon: f64,
    /// Largest delta reported against this allocation
    pub consumed_delta: f64,
    /// Allocation timestamp
    pub timestamp: u64,
    /// Expiration timestamp
    pub expires_at: Option<u64>,
}
/// Consumption pattern
#[derive(Debug, Clone)]
pub struct ConsumptionPattern {
    /// Pattern identifier (the purpose it was derived from)
    pub id: String,
    /// Time window covered, in seconds
    pub time_window: u64,
    /// Average consumption rate (epsilon per second)
    pub avg_consumption_rate: f64,
    /// Largest single spend observed
    pub peak_consumption_rate: f64,
    /// Pattern type
    pub pattern_type: PatternType,
}
/// Transition function between states
pub struct TransitionFunction<T: Float + Debug + Send + Sync + 'static> {
    /// Function name
    pub name: String,
    /// Transition logic
    pub logic: TransitionLogicFn<T>,
}
/// Dashboard layout configuration
#[derive(Debug, Clone)]
pub struct DashboardLayout {
    /// Number of columns
    pub columns: u32,
    /// Widget configurations
    pub widgets: Vec<WidgetConfig>,
}
/// Compliance violation record
#[derive(Debug, Clone)]
pub struct ComplianceViolation {
    /// Violation identifier
    pub id: String,
    /// Timestamp of violation
    pub timestamp: u64,
    /// Violated rule
    pub rule_id: String,
    /// Severity level
    pub severity: RuleSeverity,
    /// Framework violated
    pub framework: ComplianceFramework,
    /// Detailed description
    pub description: String,
    /// Remediation status
    pub remediation_status: RemediationStatus,
    /// Associated audit event
    pub audit_event_id: String,
}
/// Automated remediation action
#[derive(Debug, Clone)]
pub struct RemediationAction {
    /// Action identifier
    pub id: String,
    /// Action name
    pub name: String,
    /// Action description
    pub description: String,
    /// Action execution function
    pub execution_fn: String,
    /// Required approval level
    pub approval_level: ApprovalLevel,
}
/// Individual audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event identifier
    pub id: String,
    /// Timestamp of the event
    pub timestamp: u64,
    /// Event type
    pub event_type: AuditEventType,
    /// Actor who triggered the event
    pub actor: String,
    /// Detailed event data
    pub data: AuditEventData,
    /// Privacy parameters at time of event
    pub privacy_context: PrivacyContext,
    /// Keyed integrity tag (HMAC-SHA256), set by
    /// [`EnhancedAuditSystem::log_event`]
    pub signature: Option<Vec<u8>>,
    /// Compliance annotations
    pub compliance_annotations: HashMap<ComplianceFramework, ComplianceStatus>,
}
/// System state
#[derive(Debug, Clone)]
pub struct SystemState<T: Float + Debug + Send + Sync + 'static> {
    /// State identifier
    pub id: String,
    /// State variables
    pub variables: HashMap<String, T>,
    /// Privacy parameters
    pub privacy_params: PrivacyContext,
}
/// Types of system properties
#[derive(Debug, Clone, Copy)]
pub enum PropertyType {
    /// Safety property (something bad never happens)
    Safety,
    /// Liveness property (something good eventually happens)
    Liveness,
    /// Temporal property (time-based property)
    Temporal,
    /// Invariant property (always true)
    Invariant,
}
/// Alert status
#[derive(Debug, Clone, Copy)]
pub enum AlertStatus {
    /// Active alert
    Active,
    /// Acknowledged alert
    Acknowledged,
    /// Resolved alert
    Resolved,
    /// Suppressed alert
    Suppressed,
}
/// Reporting periods
#[derive(Debug, Clone)]
pub enum ReportingPeriod {
    /// The last 24 hours
    Daily,
    /// The last 7 days
    Weekly,
    /// The last 30 days
    Monthly,
    /// The last 91 days
    Quarterly,
    /// The last 365 days
    Annual,
    /// An explicit inclusive `[start, end]` timestamp range
    Custom(u64, u64),
}
/// Dashboard metric
#[derive(Debug, Clone)]
pub struct DashboardMetric {
    /// Metric name
    pub name: String,
    /// Current value
    pub current_value: f64,
    /// Historical values
    pub historical_values: VecDeque<(u64, f64)>,
    /// Metric unit
    pub unit: String,
    /// Metric type
    pub metric_type: MetricType,
}
/// External compliance API
///
/// Registration records the endpoint only: this crate performs no network I/O,
/// so an external API never contributes to a compliance verdict.
pub struct ExternalComplianceAPI {
    /// API name
    pub name: String,
    /// API endpoint
    pub endpoint: String,
    /// API key
    pub api_key: Option<String>,
    /// Supported frameworks
    pub frameworks: Vec<ComplianceFramework>,
}
/// Proof algorithm
pub struct ProofAlgorithm<T: Float + Debug + Send + Sync + 'static> {
    /// Algorithm name
    pub name: String,
    /// Proof generation function
    pub generate_fn: ProofGenerateFn<T>,
    /// Proof verification function
    pub verify_fn: ProofVerifyFn<T>,
}
/// Types of budget alerts
#[derive(Debug, Clone, Copy)]
pub enum BudgetAlertType {
    /// Budget nearly exhausted
    BudgetNearlyExhausted,
    /// Budget exhausted
    BudgetExhausted,
    /// Unusual consumption pattern
    UnusualConsumption,
    /// Budget allocation expired
    AllocationExpired,
    /// Budget reallocation needed
    ReallocationNeeded,
}
/// Mathematical axiom
pub struct Axiom<T: Float + Debug + Send + Sync + 'static> {
    /// Axiom name
    pub name: String,
    /// Formal statement
    pub statement: String,
    /// Axiom verification function
    pub verify_fn: AxiomVerifyFn<T>,
}
/// Configuration for audit system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Record every event type. When false, only privacy-relevant event types
    /// are recorded and the rest are skipped.
    pub comprehensive_logging: bool,
    /// Feed the monitoring dashboard from every logged event
    pub real_time_monitoring: bool,
    /// Enable formal verification. When false,
    /// [`EnhancedAuditSystem::verify_system_properties`] returns an error
    /// rather than an empty (and therefore vacuously passing) result.
    pub formal_verification: bool,
    /// Retention period for audit logs (days); must be positive
    pub retention_period_days: u32,
    /// Compliance frameworks to check
    pub compliance_frameworks: Vec<ComplianceFramework>,
    /// Cryptographic proof requirements
    pub proof_requirements: ProofRequirements,
    /// Encrypt the audit trail at rest.
    ///
    /// Not implemented: this build carries no authenticated-encryption
    /// primitive, so setting it makes [`EnhancedAuditSystem::new`] fail rather
    /// than storing plaintext behind a flag that claims otherwise.
    pub encrypt_audit_trail: bool,
    /// Submit the trail to an external auditor.
    ///
    /// Not implemented: this crate performs no network I/O, so setting it makes
    /// [`EnhancedAuditSystem::new`] fail.
    pub external_audit_integration: bool,
}
impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            comprehensive_logging: true,
            real_time_monitoring: true,
            formal_verification: true,
            retention_period_days: 365,
            compliance_frameworks: super::compliance::supported_frameworks(),
            proof_requirements: ProofRequirements::default(),
            encrypt_audit_trail: false,
            external_audit_integration: false,
        }
    }
}
/// Audit event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEventData {
    /// Event description
    pub description: String,
    /// Affected data subjects
    pub affected_data_subjects: Vec<String>,
    /// Data categories involved
    pub data_categories: Vec<String>,
    /// Processing purposes; the first is the budget purpose the event is
    /// accounted against
    pub processing_purposes: Vec<String>,
    /// Legal basis for processing
    pub legal_basis: Vec<String>,
    /// Technical measures applied
    pub technical_measures: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}
/// Result of theorem proving
#[derive(Debug, Clone)]
pub struct ProofResult {
    /// Whether theorem was proven
    pub proven: bool,
    /// Proof steps
    pub proof_steps: Vec<String>,
    /// Used axioms
    pub used_axioms: Vec<String>,
    /// Proof confidence
    pub confidence: f64,
}
/// Budget consumption record
#[derive(Debug, Clone)]
pub struct BudgetConsumption {
    /// Consumption identifier
    pub id: String,
    /// Timestamp
    pub timestamp: u64,
    /// Purpose
    pub purpose: String,
    /// Epsilon consumed
    pub epsilon_consumed: f64,
    /// Delta the spend is quoted at
    pub delta_consumed: f64,
    /// Operation performed
    pub operation: String,
}
/// Compliance report
#[derive(Debug, Clone)]
pub struct ComplianceReport {
    /// Report identifier
    pub id: String,
    /// Report timestamp
    pub timestamp: u64,
    /// Reporting period
    pub period: ReportingPeriod,
    /// Frameworks covered
    pub frameworks: Vec<ComplianceFramework>,
    /// Overall compliance status
    pub overall_status: ComplianceStatus,
    /// Detailed assessments, for frameworks that have a rule set
    pub assessments: HashMap<ComplianceFramework, ComplianceAssessment>,
    /// Executive summary
    pub executive_summary: String,
    /// Report format
    pub format: ReportFormat,
}
/// Dashboard alert
#[derive(Debug, Clone)]
pub struct DashboardAlert {
    /// Alert identifier
    pub id: String,
    /// Alert message
    pub message: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Timestamp
    pub timestamp: u64,
    /// Related metric
    pub metric: Option<String>,
    /// Alert status
    pub status: AlertStatus,
}
/// Widget configuration
#[derive(Debug, Clone)]
pub struct WidgetConfig {
    /// Widget identifier
    pub id: String,
    /// Widget type
    pub widget_type: WidgetType,
    /// Associated metrics
    pub metrics: Vec<String>,
    /// Widget position
    pub position: (u32, u32),
    /// Widget size
    pub size: (u32, u32),
}
/// Compliance rule evaluation result
#[derive(Debug, Clone)]
pub struct ComplianceRuleResult {
    /// Whether rule passed
    pub passed: bool,
    /// Detailed result message
    pub message: String,
    /// Recommended actions
    pub recommendations: Vec<String>,
    /// Risk level
    pub risk_level: RiskLevel,
}
/// Types of dashboard widgets
#[derive(Debug, Clone, Copy)]
pub enum WidgetType {
    /// Line chart
    LineChart,
    /// Bar chart
    BarChart,
    /// Pie chart
    PieChart,
    /// Gauge widget
    Gauge,
    /// Table widget
    Table,
    /// Text display
    Text,
    /// Alert list
    AlertList,
}
/// Cryptographic proof type
pub struct CryptographicProofType<T: Float + Debug + Send + Sync + 'static> {
    /// Proof type name
    pub name: String,
    /// Proof generation function
    pub generate_fn: CryptoProofGenerateFn<T>,
    /// Proof verification function
    pub verify_fn: CryptoProofVerifyFn<T>,
}
/// Cryptographic proof
#[derive(Debug, Clone)]
pub struct CryptographicProof {
    /// Proof type
    pub prooftype: String,
    /// Proof data
    pub proof_data: Vec<u8>,
    /// Public parameters
    pub public_params: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
    /// Proof metadata
    pub metadata: HashMap<String, String>,
}
/// Compliance rule definition
pub struct ComplianceRule {
    /// Rule identifier
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule evaluation function
    pub evaluation_fn: Box<dyn Fn(&AuditEvent) -> ComplianceRuleResult + Send + Sync>,
    /// Rule severity
    pub severity: RuleSeverity,
    /// Applicable frameworks
    pub frameworks: Vec<ComplianceFramework>,
}

/// Whether an event type is privacy-relevant.
///
/// Used to honour `AuditConfig::comprehensive_logging`: with comprehensive
/// logging off, only these event types are recorded.
fn is_privacy_relevant(event_type: &AuditEventType) -> bool {
    matches!(
        event_type,
        AuditEventType::PrivacyBudgetAllocation
            | AuditEventType::PrivacyBudgetConsumption
            | AuditEventType::GradientComputation
            | AuditEventType::DataAccess
            | AuditEventType::DataDeletion
            | AuditEventType::UserConsent
            | AuditEventType::AnonymizationProcess
            | AuditEventType::SecurityIncident
    )
}

/// Enhanced audit system for privacy-preserving optimization.
pub struct EnhancedAuditSystem<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration for audit system
    config: AuditConfig,
    /// Audit trail storage
    audit_trail: AuditTrail,
    /// Compliance monitor
    compliance_monitor: ComplianceMonitor,
    /// Formal verification engine
    verification_engine: FormalVerificationEngine<T>,
    /// Privacy budget tracker
    privacy_tracker: PrivacyBudgetTracker,
    /// Cryptographic proof generator
    proof_generator: CryptographicProofGenerator<T>,
    /// Regulatory compliance checker
    regulatory_checker: RegulatoryComplianceChecker,
    /// Real-time monitoring dashboard
    monitoring_dashboard: MonitoringDashboard,
    /// Key used for the per-event integrity tag
    event_tag_key: Digest32,
    /// Number of events skipped because comprehensive logging is off
    skipped_events: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> EnhancedAuditSystem<T> {
    /// Create a new enhanced audit system.
    ///
    /// Returns an error when the configuration asks for a guarantee this build
    /// cannot provide (at-rest encryption, external audit submission, or a
    /// proof requirement outside the integrity family). Accepting such a
    /// configuration and quietly not honouring it is exactly the failure mode
    /// this module is being repaired for, so the constructor is fallible.
    pub fn new(config: AuditConfig) -> Result<Self> {
        if config.retention_period_days == 0 {
            return Err(OptimError::InvalidConfig(
                "retention_period_days must be positive".to_string(),
            ));
        }
        if config.encrypt_audit_trail {
            return Err(OptimError::UnsupportedOperation(
                "encrypt_audit_trail was requested, but this build carries no \
                 authenticated-encryption primitive; the trail would be stored in plaintext"
                    .to_string(),
            ));
        }
        if config.external_audit_integration {
            return Err(OptimError::UnsupportedOperation(
                "external_audit_integration was requested, but this crate performs no network I/O"
                    .to_string(),
            ));
        }

        let proof_generator = CryptographicProofGenerator::<T>::new();
        proof_generator.check_requirements(&config.proof_requirements)?;

        let compliance_monitor = ComplianceMonitor::for_frameworks(&config.compliance_frameworks);
        let verification_engine = if config.formal_verification {
            FormalVerificationEngine::<T>::new()
        } else {
            FormalVerificationEngine::<T>::empty()
        };

        Ok(Self {
            config,
            audit_trail: AuditTrail::new(),
            compliance_monitor,
            verification_engine,
            privacy_tracker: PrivacyBudgetTracker::new(),
            proof_generator,
            regulatory_checker: RegulatoryComplianceChecker::new(),
            monitoring_dashboard: MonitoringDashboard::new(),
            event_tag_key: random_key(),
            skipped_events: 0,
        })
    }

    /// The active configuration.
    pub fn config(&self) -> &AuditConfig {
        &self.config
    }

    /// Read-only access to the audit trail.
    pub fn audit_trail(&self) -> &AuditTrail {
        &self.audit_trail
    }

    /// Read-only access to the compliance monitor.
    pub fn compliance_monitor(&self) -> &ComplianceMonitor {
        &self.compliance_monitor
    }

    /// Read-only access to the monitoring dashboard.
    pub fn monitoring_dashboard(&self) -> &MonitoringDashboard {
        &self.monitoring_dashboard
    }

    /// Read-only access to the privacy budget tracker.
    pub fn privacy_tracker(&self) -> &PrivacyBudgetTracker {
        &self.privacy_tracker
    }

    /// Read-only access to the formal verification engine.
    pub fn verification_engine(&self) -> &FormalVerificationEngine<T> {
        &self.verification_engine
    }

    /// Number of events skipped because comprehensive logging is off.
    pub fn skipped_event_count(&self) -> usize {
        self.skipped_events
    }

    /// Allocate privacy budget to a processing purpose.
    pub fn allocate_privacy_budget(
        &mut self,
        purpose: impl Into<String>,
        epsilon: f64,
        delta: f64,
        timestamp: u64,
        expires_at: Option<u64>,
    ) -> Result<()> {
        self.privacy_tracker
            .allocate(purpose, epsilon, delta, timestamp, expires_at)
    }

    /// Log an audit event.
    ///
    /// Returns the number of compliance violations the event triggered, so a
    /// caller can react instead of having them written to stderr.
    pub fn log_event(&mut self, event: AuditEvent) -> Result<usize> {
        if !self.config.comprehensive_logging && !is_privacy_relevant(&event.event_type) {
            self.skipped_events += 1;
            return Ok(0);
        }

        let signed_event = self.sign_event(event);
        self.audit_trail.add_event(signed_event.clone())?;
        let violations = self.compliance_monitor.check_event(&signed_event)?;
        if matches!(
            signed_event.event_type,
            AuditEventType::PrivacyBudgetConsumption
        ) {
            self.privacy_tracker.record_consumption(&signed_event)?;
        }
        if self.config.real_time_monitoring {
            self.monitoring_dashboard.update_metrics(&signed_event)?;
        }
        Ok(violations)
    }

    /// Re-derive every commitment in the audit trail.
    pub fn verify_audit_trail(&self) -> Result<()> {
        self.audit_trail.verify_integrity()
    }

    /// Verify the integrity tag of a stored event.
    pub fn verify_event_tag(&self, event: &AuditEvent) -> bool {
        let Some(recorded) = event.signature.as_ref() else {
            return false;
        };
        let expected = self.compute_event_tag(event);
        recorded.len() == expected.len() && {
            let mut difference = 0u8;
            for (left, right) in recorded.iter().zip(expected.iter()) {
                difference |= left ^ right;
            }
            difference == 0
        }
    }

    /// Query the audit trail.
    pub fn query_events(&self, criteria: &AuditQueryCriteria) -> Vec<&AuditEvent> {
        self.audit_trail.query_events(criteria)
    }

    /// Identifiers of events whose age exceeds the retention period.
    ///
    /// The trail is append-only and tamper-evident: deleting an event in place
    /// would break the hash chain and make every later verification fail. This
    /// returns the events a caller may export and then discard *with the whole
    /// trail*, rather than pretending in-place pruning is possible.
    pub fn events_past_retention(&self, now: u64) -> Vec<String> {
        let retention_seconds = u64::from(self.config.retention_period_days) * 86_400;
        let cutoff = now.saturating_sub(retention_seconds);
        self.audit_trail
            .events()
            .filter(|event| event.timestamp < cutoff)
            .map(|event| event.id.clone())
            .collect()
    }

    /// Generate a compliance report from the recorded events.
    pub fn generate_compliance_report(
        &self,
        frameworks: &[ComplianceFramework],
        period: ReportingPeriod,
    ) -> Result<ComplianceReport> {
        self.regulatory_checker
            .generate_report(frameworks, period, &self.audit_trail)
    }

    /// Verify the formal properties of a release.
    pub fn verify_system_properties(
        &self,
        data: &Array1<T>,
        context: &PrivacyContext,
    ) -> Result<Vec<VerificationResult>> {
        if !self.config.formal_verification {
            return Err(OptimError::InvalidState(
                "formal_verification is disabled in the audit configuration, so no property was \
                 checked"
                    .to_string(),
            ));
        }
        self.verification_engine
            .verify_all_properties(data, context)
    }

    /// Verify the formal properties of a release, failing on any critical
    /// violation.
    pub fn require_system_properties(
        &self,
        data: &Array1<T>,
        context: &PrivacyContext,
    ) -> Result<Vec<VerificationResult>> {
        if !self.config.formal_verification {
            return Err(OptimError::InvalidState(
                "formal_verification is disabled in the audit configuration".to_string(),
            ));
        }
        self.verification_engine
            .require_all_properties(data, context)
    }

    /// Check a bounded invariant property of the registered system model.
    pub fn check_model_property(&self, property: &SystemProperty) -> Result<ModelCheckOutcome> {
        if !self.config.formal_verification {
            return Err(OptimError::InvalidState(
                "formal_verification is disabled in the audit configuration".to_string(),
            ));
        }
        self.verification_engine.check_model_property(property)
    }

    /// Current privacy budget status, by purpose.
    pub fn get_privacy_budget_status(&self) -> HashMap<String, BudgetAllocation> {
        self.privacy_tracker.get_current_allocations()
    }

    /// Generate a cryptographic proof over a released vector.
    pub fn generate_proof(&self, prooftype: &str, data: &Array1<T>) -> Result<CryptographicProof> {
        self.proof_generator.generate_proof(prooftype, data)
    }

    /// Generate the default (unkeyed) integrity proof.
    pub fn generate_integrity_proof(&self, data: &Array1<T>) -> Result<CryptographicProof> {
        self.proof_generator.generate_proof(SHA256_INTEGRITY, data)
    }

    /// Verify a cryptographic proof against a released vector.
    pub fn verify_proof(&self, proof: &CryptographicProof, data: &Array1<T>) -> Result<bool> {
        self.proof_generator.verify_proof(proof, data)
    }

    /// Attach the integrity tag to an event.
    fn sign_event(&self, mut event: AuditEvent) -> AuditEvent {
        event.signature = None;
        let tag = self.compute_event_tag(&event);
        event.signature = Some(tag);
        event
    }

    /// HMAC-SHA256 over the canonical encoding of an event.
    ///
    /// The previous implementation hashed four fields with `serde_json` and
    /// `.expect("unwrap failed")`, so most of the event was unauthenticated and
    /// a serialisation failure was a panic.
    fn compute_event_tag(&self, event: &AuditEvent) -> Vec<u8> {
        hmac_sha256(&self.event_tag_key, &canonical_event_bytes(event)).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(id: &str, epsilon: f64, timestamp: u64) -> AuditEvent {
        AuditEvent {
            id: id.to_string(),
            timestamp,
            event_type: AuditEventType::PrivacyBudgetConsumption,
            actor: "trainer".to_string(),
            data: AuditEventData {
                description: "spend".to_string(),
                affected_data_subjects: Vec::new(),
                data_categories: Vec::new(),
                processing_purposes: vec!["ml_training".to_string()],
                legal_basis: vec!["consent".to_string()],
                technical_measures: vec!["differential_privacy".to_string()],
                metadata: HashMap::new(),
            },
            privacy_context: PrivacyContext {
                epsilon_budget: epsilon,
                delta_budget: 1e-6,
                privacy_mechanism: "dp_sgd".to_string(),
                data_minimization: true,
                purpose_limitation: true,
                storage_limitation: true,
            },
            signature: None,
            compliance_annotations: HashMap::new(),
        }
    }

    fn system() -> EnhancedAuditSystem<f64> {
        match EnhancedAuditSystem::<f64>::new(AuditConfig::default()) {
            Ok(system) => system,
            Err(err) => panic!("construction failed: {err}"),
        }
    }

    #[test]
    fn logging_events_drives_the_trail_the_budget_and_the_dashboard() {
        let mut audit = system();
        let ok = audit.allocate_privacy_budget("ml_training", 1.0, 1e-5, 0, None);
        assert!(ok.is_ok());

        for step in 0..3u64 {
            let violations = match audit.log_event(event(&format!("e{step}"), 0.1, 100 + step)) {
                Ok(count) => count,
                Err(err) => panic!("log_event failed: {err}"),
            };
            assert_eq!(violations, 0, "a well-formed event violates nothing");
        }

        assert_eq!(audit.audit_trail().len(), 3);
        assert!(audit.verify_audit_trail().is_ok());

        let status = audit.get_privacy_budget_status();
        let training = match status.get("ml_training") {
            Some(allocation) => allocation,
            None => panic!("the allocation must be reported"),
        };
        assert!((training.consumed_epsilon - 0.3).abs() < 1e-12);

        let events_metric = audit
            .monitoring_dashboard()
            .metric(super::super::dashboard::METRIC_EVENTS)
            .map(|metric| metric.current_value);
        assert_eq!(events_metric, Some(3.0));
    }

    #[test]
    fn every_logged_event_carries_a_verifiable_integrity_tag() {
        let mut audit = system();
        let ok = audit.log_event(event("e0", 0.1, 100));
        assert!(ok.is_ok());
        let stored = match audit.audit_trail().events().next() {
            Some(stored) => stored.clone(),
            None => panic!("the event must be stored"),
        };
        assert!(stored.signature.is_some());
        assert!(audit.verify_event_tag(&stored));

        let mut tampered = stored;
        tampered.privacy_context.epsilon_budget = 9.0;
        assert!(
            !audit.verify_event_tag(&tampered),
            "the tag must cover the privacy context"
        );
    }

    #[test]
    fn a_configuration_asking_for_at_rest_encryption_is_refused() {
        let config = AuditConfig {
            encrypt_audit_trail: true,
            ..AuditConfig::default()
        };
        let outcome = EnhancedAuditSystem::<f64>::new(config);
        let message = match outcome {
            Err(err) => err.to_string(),
            Ok(_) => panic!("an unimplementable guarantee must not be accepted"),
        };
        assert!(
            message.contains("authenticated-encryption"),
            "got: {message}"
        );
    }

    #[test]
    fn a_configuration_asking_for_external_audit_submission_is_refused() {
        let config = AuditConfig {
            external_audit_integration: true,
            ..AuditConfig::default()
        };
        assert!(EnhancedAuditSystem::<f64>::new(config).is_err());
    }

    #[test]
    fn a_configuration_asking_for_zero_knowledge_proofs_is_refused() {
        let config = AuditConfig {
            proof_requirements: ProofRequirements {
                zero_knowledge_proofs: true,
                ..ProofRequirements::default()
            },
            ..AuditConfig::default()
        };
        assert!(EnhancedAuditSystem::<f64>::new(config).is_err());
    }

    #[test]
    fn a_zero_retention_period_is_refused() {
        let config = AuditConfig {
            retention_period_days: 0,
            ..AuditConfig::default()
        };
        assert!(EnhancedAuditSystem::<f64>::new(config).is_err());
    }

    #[test]
    fn disabling_formal_verification_errors_instead_of_passing_vacuously() {
        // The old engine returned Ok(vec![]) with no rules registered, which
        // reads as "verified". With verification disabled the call must fail.
        let config = AuditConfig {
            formal_verification: false,
            ..AuditConfig::default()
        };
        let audit = match EnhancedAuditSystem::<f64>::new(config) {
            Ok(audit) => audit,
            Err(err) => panic!("construction failed: {err}"),
        };
        let data = Array1::from(vec![0.1, 0.2]);
        let context = event("e", 0.1, 1).privacy_context;
        assert!(audit.verify_system_properties(&data, &context).is_err());
    }

    #[test]
    fn formal_verification_runs_real_rules_by_default() {
        let audit = system();
        let context = event("e", 0.1, 1).privacy_context;
        let results = match audit.verify_system_properties(&Array1::from(vec![0.1, 0.2]), &context)
        {
            Ok(results) => results,
            Err(err) => panic!("verification failed: {err}"),
        };
        assert!(
            !results.is_empty(),
            "the default rule set must not be empty"
        );
        assert!(results.iter().all(|result| result.verified));

        let bad = Array1::from(vec![f64::NAN]);
        assert!(audit.require_system_properties(&bad, &context).is_err());
    }

    #[test]
    fn comprehensive_logging_off_skips_non_privacy_events() {
        let config = AuditConfig {
            comprehensive_logging: false,
            ..AuditConfig::default()
        };
        let mut audit = match EnhancedAuditSystem::<f64>::new(config) {
            Ok(audit) => audit,
            Err(err) => panic!("construction failed: {err}"),
        };
        let mut chatter = event("noise", 0.0, 10);
        chatter.event_type = AuditEventType::SystemLifecycle;
        let ok = audit.log_event(chatter);
        assert!(ok.is_ok());
        assert_eq!(audit.audit_trail().len(), 0);
        assert_eq!(audit.skipped_event_count(), 1);

        let ok = audit.log_event(event("spend", 0.1, 11));
        assert!(ok.is_ok());
        assert_eq!(audit.audit_trail().len(), 1);
    }

    #[test]
    fn a_violating_event_is_reported_to_the_caller() {
        let mut audit = system();
        let mut bad = event("bad", 0.1, 100);
        bad.data.legal_basis.clear();
        let violations = match audit.log_event(bad) {
            Ok(count) => count,
            Err(err) => panic!("log_event failed: {err}"),
        };
        assert!(violations > 0, "a missing legal basis is a GDPR violation");
        assert_eq!(audit.compliance_monitor().violation_count(), violations);
    }

    #[test]
    fn the_compliance_report_reflects_the_recorded_events() {
        let mut audit = system();
        let now = super::super::proofs::unix_timestamp().unwrap_or_default();
        let ok = audit.log_event(event("good", 0.1, now));
        assert!(ok.is_ok());
        let report = match audit
            .generate_compliance_report(&[ComplianceFramework::GDPR], ReportingPeriod::Daily)
        {
            Ok(report) => report,
            Err(err) => panic!("report failed: {err}"),
        };
        assert!(matches!(report.overall_status, ComplianceStatus::Compliant));
        assert!(report.assessments.contains_key(&ComplianceFramework::GDPR));
    }

    #[test]
    fn proofs_generated_by_the_system_verify() {
        let audit = system();
        let data = Array1::from(vec![0.5, -0.25]);
        let proof = match audit.generate_integrity_proof(&data) {
            Ok(proof) => proof,
            Err(err) => panic!("proof failed: {err}"),
        };
        match audit.verify_proof(&proof, &data) {
            Ok(true) => {}
            Ok(false) => panic!("the proof must verify"),
            Err(err) => panic!("verification failed: {err}"),
        }
        match audit.verify_proof(&proof, &Array1::from(vec![0.5, -0.26])) {
            Ok(false) => {}
            Ok(true) => panic!("the proof must not verify against other data"),
            Err(err) => panic!("verification failed: {err}"),
        }
    }

    #[test]
    fn events_past_retention_are_listed_rather_than_silently_pruned() {
        let config = AuditConfig {
            retention_period_days: 1,
            ..AuditConfig::default()
        };
        let mut audit = match EnhancedAuditSystem::<f64>::new(config) {
            Ok(audit) => audit,
            Err(err) => panic!("construction failed: {err}"),
        };
        let ok = audit.log_event(event("old", 0.1, 0));
        assert!(ok.is_ok());
        let ok = audit.log_event(event("new", 0.1, 200_000));
        assert!(ok.is_ok());

        let expired = audit.events_past_retention(200_000);
        assert_eq!(expired, vec!["old".to_string()]);
        // Pruning is not performed, so the chain still verifies.
        assert!(audit.verify_audit_trail().is_ok());
        assert_eq!(audit.audit_trail().len(), 2);
    }

    #[test]
    fn queries_filter_by_actor_and_time() {
        let mut audit = system();
        let mut first = event("a", 0.1, 100);
        first.actor = "alice".to_string();
        let mut second = event("b", 0.1, 200);
        second.actor = "bob".to_string();
        let ok = audit.log_event(first);
        assert!(ok.is_ok());
        let ok = audit.log_event(second);
        assert!(ok.is_ok());

        let alice = audit.query_events(&AuditQueryCriteria {
            actor: Some("alice".to_string()),
            ..AuditQueryCriteria::any()
        });
        assert_eq!(alice.len(), 1);
        let late = audit.query_events(&AuditQueryCriteria {
            start_time: Some(150),
            ..AuditQueryCriteria::any()
        });
        assert_eq!(late.len(), 1);
        assert_eq!(late[0].id, "b");
    }
}
