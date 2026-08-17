//! Regulatory compliance checking driven by real rule sets.
//!
//! # The defect this replaces
//!
//! `RegulatoryComplianceChecker::new` created an empty `regulations` map that
//! nothing ever populated, and `generate_report` ignored the audit trail
//! entirely, returning a hardcoded
//!
//! ```text
//! overall_status: ComplianceStatus::Compliant,
//! assessments: HashMap::new(),
//! executive_summary: "Compliance report generated successfully",
//! ```
//!
//! for any input. A GDPR/HIPAA compliance verdict that is a compile-time
//! constant is the most consequential lie in this module.
//!
//! `ComplianceMonitor::check_event` had the mirror-image problem: it iterated
//! an always-empty `frameworks` vector, and on the impossible branch where a
//! rule did fire it wrote to stderr instead of recording a violation (the
//! `record_violation` method existed and was never called).
//!
//! # What is implemented
//!
//! * Rule sets for GDPR, HIPAA and CCPA, each rule an exact predicate over the
//!   recorded [`AuditEvent`] and citing the article it derives from.
//! * `generate_report` evaluates every rule against every event inside the
//!   reporting period and derives the status, score and risk from the actual
//!   findings.
//! * **`Compliant` is never the default.** A framework with no rule set, or a
//!   period with no events, yields
//!   [`ComplianceStatus::RequiresReview`] -- absence of evidence is not
//!   evidence of compliance. Any critical or high-severity violation yields
//!   [`ComplianceStatus::NonCompliant`].

use crate::error::Result;
use std::collections::{HashMap, VecDeque};

use super::hashing::framework_key;
use super::integrity::AuditTrail;
use super::proofs::unix_timestamp;
use super::types::{
    AuditEvent, AuditEventType, ComplianceAssessment, ComplianceFinding, ComplianceFramework,
    ComplianceReport, ComplianceRule, ComplianceRuleResult, ComplianceStatus, ComplianceViolation,
    ExternalComplianceAPI, ImpactLevel, RemediationAction, RemediationStatus, ReportFormat,
    ReportingPeriod, RiskAssessment, RiskFactor, RiskLevel, RuleSeverity,
};

/// Weight used to turn a rule severity into a risk contribution.
fn severity_weight(severity: RuleSeverity) -> f64 {
    match severity {
        RuleSeverity::Critical => 1.0,
        RuleSeverity::High => 0.7,
        RuleSeverity::Medium => 0.4,
        RuleSeverity::Low => 0.2,
        RuleSeverity::Info => 0.05,
    }
}

/// Impact level implied by a rule severity.
fn severity_impact(severity: RuleSeverity) -> ImpactLevel {
    match severity {
        RuleSeverity::Critical => ImpactLevel::VeryHigh,
        RuleSeverity::High => ImpactLevel::High,
        RuleSeverity::Medium => ImpactLevel::Medium,
        RuleSeverity::Low => ImpactLevel::Low,
        RuleSeverity::Info => ImpactLevel::VeryLow,
    }
}

/// Risk level implied by a rule severity.
fn severity_risk(severity: RuleSeverity) -> RiskLevel {
    match severity {
        RuleSeverity::Critical => RiskLevel::VeryHigh,
        RuleSeverity::High => RiskLevel::High,
        RuleSeverity::Medium => RiskLevel::Medium,
        RuleSeverity::Low => RiskLevel::Low,
        RuleSeverity::Info => RiskLevel::VeryLow,
    }
}

/// A passing rule result.
fn passed(message: impl Into<String>) -> ComplianceRuleResult {
    ComplianceRuleResult {
        passed: true,
        message: message.into(),
        recommendations: Vec::new(),
        risk_level: RiskLevel::VeryLow,
    }
}

/// A failing rule result.
fn failed(
    severity: RuleSeverity,
    message: impl Into<String>,
    recommendation: impl Into<String>,
) -> ComplianceRuleResult {
    ComplianceRuleResult {
        passed: false,
        message: message.into(),
        recommendations: vec![recommendation.into()],
        risk_level: severity_risk(severity),
    }
}

/// GDPR rule set (Regulation (EU) 2016/679).
pub fn gdpr_rules() -> Vec<ComplianceRule> {
    vec![
        ComplianceRule {
            id: "gdpr-art6-legal-basis".to_string(),
            name: "Art. 6 lawfulness of processing".to_string(),
            description: "Every processing operation records at least one legal basis.".to_string(),
            severity: RuleSeverity::Critical,
            frameworks: vec![ComplianceFramework::GDPR],
            evaluation_fn: Box::new(|event: &AuditEvent| {
                if event.data.legal_basis.is_empty() {
                    failed(
                        RuleSeverity::Critical,
                        format!("event `{}` records no legal basis for processing", event.id),
                        "record the Art. 6(1) legal basis on every processing event",
                    )
                } else {
                    passed(format!(
                        "legal basis: {}",
                        event.data.legal_basis.join(", ")
                    ))
                }
            }),
        },
        ComplianceRule {
            id: "gdpr-art5-1b-purpose-limitation".to_string(),
            name: "Art. 5(1)(b) purpose limitation".to_string(),
            description: "Processing purposes are declared and purpose limitation is asserted."
                .to_string(),
            severity: RuleSeverity::High,
            frameworks: vec![ComplianceFramework::GDPR],
            evaluation_fn: Box::new(|event: &AuditEvent| {
                if event.data.processing_purposes.is_empty() {
                    failed(
                        RuleSeverity::High,
                        format!("event `{}` declares no processing purpose", event.id),
                        "declare the specified, explicit and legitimate purpose",
                    )
                } else if !event.privacy_context.purpose_limitation {
                    failed(
                        RuleSeverity::High,
                        format!("event `{}` does not assert purpose limitation", event.id),
                        "set purpose_limitation once processing is confined to the declared purpose",
                    )
                } else {
                    passed(format!(
                        "purposes: {}",
                        event.data.processing_purposes.join(", ")
                    ))
                }
            }),
        },
        ComplianceRule {
            id: "gdpr-art5-1c-data-minimisation".to_string(),
            name: "Art. 5(1)(c) data minimisation".to_string(),
            description: "Data minimisation is asserted for the processing operation.".to_string(),
            severity: RuleSeverity::High,
            frameworks: vec![ComplianceFramework::GDPR],
            evaluation_fn: Box::new(|event: &AuditEvent| {
                if event.privacy_context.data_minimization {
                    passed("data minimisation asserted")
                } else {
                    failed(
                        RuleSeverity::High,
                        format!("event `{}` does not assert data minimisation", event.id),
                        "restrict the processed fields to what the declared purpose requires",
                    )
                }
            }),
        },
        ComplianceRule {
            id: "gdpr-art5-1e-storage-limitation".to_string(),
            name: "Art. 5(1)(e) storage limitation".to_string(),
            description: "Storage limitation is asserted for the processing operation.".to_string(),
            severity: RuleSeverity::Medium,
            frameworks: vec![ComplianceFramework::GDPR],
            evaluation_fn: Box::new(|event: &AuditEvent| {
                if event.privacy_context.storage_limitation {
                    passed("storage limitation asserted")
                } else {
                    failed(
                        RuleSeverity::Medium,
                        format!("event `{}` does not assert storage limitation", event.id),
                        "attach a retention period to the processed data",
                    )
                }
            }),
        },
        ComplianceRule {
            id: "gdpr-art32-technical-measures".to_string(),
            name: "Art. 32 security of processing".to_string(),
            description: "At least one technical measure is recorded.".to_string(),
            severity: RuleSeverity::High,
            frameworks: vec![ComplianceFramework::GDPR],
            evaluation_fn: Box::new(|event: &AuditEvent| {
                if event.data.technical_measures.is_empty() {
                    failed(
                        RuleSeverity::High,
                        format!("event `{}` records no technical measure", event.id),
                        "record the applied technical and organisational measures",
                    )
                } else {
                    passed(format!(
                        "measures: {}",
                        event.data.technical_measures.join(", ")
                    ))
                }
            }),
        },
        ComplianceRule {
            id: "gdpr-privacy-budget-is-meaningful".to_string(),
            name: "Privacy budget is a guarantee".to_string(),
            description: "The recorded epsilon is positive and finite and delta lies in [0, 1)."
                .to_string(),
            severity: RuleSeverity::Critical,
            frameworks: vec![ComplianceFramework::GDPR],
            evaluation_fn: Box::new(|event: &AuditEvent| {
                let epsilon = event.privacy_context.epsilon_budget;
                let delta = event.privacy_context.delta_budget;
                if !epsilon.is_finite() || epsilon <= 0.0 {
                    failed(
                        RuleSeverity::Critical,
                        format!("event `{}` records epsilon = {epsilon}", event.id),
                        "record the composed epsilon actually enforced for this release",
                    )
                } else if !delta.is_finite() || !(0.0..1.0).contains(&delta) {
                    failed(
                        RuleSeverity::Critical,
                        format!("event `{}` records delta = {delta}", event.id),
                        "record a delta in [0, 1)",
                    )
                } else {
                    passed(format!("epsilon = {epsilon}, delta = {delta}"))
                }
            }),
        },
    ]
}

/// HIPAA rule set (45 CFR Part 164).
pub fn hipaa_rules() -> Vec<ComplianceRule> {
    vec![
        ComplianceRule {
            id: "hipaa-164-312-a-access-control".to_string(),
            name: "164.312(a) access control".to_string(),
            description: "Every event identifies the acting principal.".to_string(),
            severity: RuleSeverity::Critical,
            frameworks: vec![ComplianceFramework::HIPAA],
            evaluation_fn: Box::new(|event: &AuditEvent| {
                if event.actor.trim().is_empty() {
                    failed(
                        RuleSeverity::Critical,
                        format!("event `{}` has no identified actor", event.id),
                        "record the authenticated principal on every event",
                    )
                } else {
                    passed(format!("actor: {}", event.actor))
                }
            }),
        },
        ComplianceRule {
            id: "hipaa-164-312-b-audit-controls".to_string(),
            name: "164.312(b) audit controls".to_string(),
            description: "Every event carries an integrity tag.".to_string(),
            severity: RuleSeverity::High,
            frameworks: vec![ComplianceFramework::HIPAA],
            evaluation_fn: Box::new(|event: &AuditEvent| match event.signature.as_ref() {
                Some(signature) if !signature.is_empty() => {
                    passed(format!("{}-byte integrity tag present", signature.len()))
                }
                _ => failed(
                    RuleSeverity::High,
                    format!("event `{}` carries no integrity tag", event.id),
                    "log events through EnhancedAuditSystem::log_event so they are tagged",
                ),
            }),
        },
        ComplianceRule {
            id: "hipaa-164-514-b-de-identification".to_string(),
            name: "164.514(b) de-identification".to_string(),
            description: "A de-identification measure is recorded for data-bearing events."
                .to_string(),
            severity: RuleSeverity::Critical,
            frameworks: vec![ComplianceFramework::HIPAA],
            evaluation_fn: Box::new(|event: &AuditEvent| {
                let recognised = [
                    "differential_privacy",
                    "de_identification",
                    "deidentification",
                    "anonymization",
                    "anonymisation",
                    "pseudonymization",
                    "pseudonymisation",
                    "aggregation",
                ];
                let applies = matches!(
                    event.event_type,
                    AuditEventType::DataAccess
                        | AuditEventType::GradientComputation
                        | AuditEventType::ModelParameterUpdate
                        | AuditEventType::AnonymizationProcess
                );
                if !applies {
                    return passed(
                        "no protected health information is released by this event type",
                    );
                }
                let has_measure = event.data.technical_measures.iter().any(|measure| {
                    let lowered = measure.to_lowercase();
                    recognised.iter().any(|known| lowered.contains(known))
                });
                if has_measure {
                    passed("a recognised de-identification measure is recorded")
                } else {
                    failed(
                        RuleSeverity::Critical,
                        format!(
                            "event `{}` releases data without a recognised de-identification \
                             measure (recorded: {:?})",
                            event.id, event.data.technical_measures
                        ),
                        "apply and record a de-identification measure before releasing data",
                    )
                }
            }),
        },
    ]
}

/// CCPA rule set (Cal. Civ. Code 1798.100 et seq.).
pub fn ccpa_rules() -> Vec<ComplianceRule> {
    vec![
        ComplianceRule {
            id: "ccpa-1798-100-purpose-disclosure".to_string(),
            name: "1798.100 notice at collection".to_string(),
            description: "The business purpose of the processing is disclosed.".to_string(),
            severity: RuleSeverity::High,
            frameworks: vec![ComplianceFramework::CCPA],
            evaluation_fn: Box::new(|event: &AuditEvent| {
                if event.data.processing_purposes.is_empty() {
                    failed(
                        RuleSeverity::High,
                        format!("event `{}` discloses no business purpose", event.id),
                        "disclose the business purpose for each category of data collected",
                    )
                } else {
                    passed("business purpose disclosed")
                }
            }),
        },
        ComplianceRule {
            id: "ccpa-1798-105-deletion-records".to_string(),
            name: "1798.105 right to delete".to_string(),
            description: "Deletion events identify the data subjects affected.".to_string(),
            severity: RuleSeverity::Medium,
            frameworks: vec![ComplianceFramework::CCPA],
            evaluation_fn: Box::new(|event: &AuditEvent| {
                if !matches!(event.event_type, AuditEventType::DataDeletion) {
                    return passed("not a deletion event");
                }
                if event.data.affected_data_subjects.is_empty() {
                    failed(
                        RuleSeverity::Medium,
                        format!(
                            "deletion event `{}` does not identify the affected data subjects",
                            event.id
                        ),
                        "record which consumer requests each deletion satisfies",
                    )
                } else {
                    passed(format!(
                        "{} data subject(s) recorded",
                        event.data.affected_data_subjects.len()
                    ))
                }
            }),
        },
        ComplianceRule {
            id: "ccpa-1798-100-data-categories".to_string(),
            name: "1798.100 categories of personal information".to_string(),
            description: "Data-bearing events record the categories processed.".to_string(),
            severity: RuleSeverity::Medium,
            frameworks: vec![ComplianceFramework::CCPA],
            evaluation_fn: Box::new(|event: &AuditEvent| {
                let applies = matches!(
                    event.event_type,
                    AuditEventType::DataAccess
                        | AuditEventType::DataDeletion
                        | AuditEventType::AnonymizationProcess
                );
                if !applies {
                    return passed("no personal information category applies");
                }
                if event.data.data_categories.is_empty() {
                    failed(
                        RuleSeverity::Medium,
                        format!("event `{}` records no data category", event.id),
                        "record the categories of personal information processed",
                    )
                } else {
                    passed(format!(
                        "categories: {}",
                        event.data.data_categories.join(", ")
                    ))
                }
            }),
        },
    ]
}

/// The rule set for a framework, or `None` when this build has none.
pub fn rules_for(framework: &ComplianceFramework) -> Option<Vec<ComplianceRule>> {
    match framework {
        ComplianceFramework::GDPR => Some(gdpr_rules()),
        ComplianceFramework::HIPAA => Some(hipaa_rules()),
        ComplianceFramework::CCPA => Some(ccpa_rules()),
        // No rule set has been written for these frameworks. Reporting them as
        // `Compliant` would be a fabrication, so they surface as
        // `RequiresReview` instead.
        ComplianceFramework::SOX
        | ComplianceFramework::FISMA
        | ComplianceFramework::ISO27001
        | ComplianceFramework::NISTPrivacy
        | ComplianceFramework::Custom(_) => None,
    }
}

/// Frameworks this build can actually assess.
pub fn supported_frameworks() -> Vec<ComplianceFramework> {
    vec![
        ComplianceFramework::GDPR,
        ComplianceFramework::HIPAA,
        ComplianceFramework::CCPA,
    ]
}

/// Rule checker for one framework.
pub struct RegulationChecker {
    /// Framework this checker covers.
    pub framework: ComplianceFramework,
    /// Rules to evaluate.
    pub rules: Vec<ComplianceRule>,
}

impl RegulationChecker {
    /// Build the checker for a framework, if a rule set exists.
    pub fn for_framework(framework: ComplianceFramework) -> Option<Self> {
        rules_for(&framework).map(|rules| Self { framework, rules })
    }
}

/// Regulatory compliance checker.
pub struct RegulatoryComplianceChecker {
    /// Rule checkers by framework key.
    regulations: HashMap<String, RegulationChecker>,
    /// Reports produced so far.
    reports: VecDeque<ComplianceReport>,
    /// Registered external compliance APIs, by name.
    external_apis: HashMap<String, ExternalComplianceAPI>,
}

impl RegulatoryComplianceChecker {
    /// Create a checker with every implemented rule set registered.
    pub fn new() -> Self {
        let mut regulations = HashMap::new();
        for framework in supported_frameworks() {
            if let Some(checker) = RegulationChecker::for_framework(framework.clone()) {
                regulations.insert(framework_key(&framework), checker);
            }
        }
        Self {
            regulations,
            reports: VecDeque::new(),
            external_apis: HashMap::new(),
        }
    }

    /// Create a checker with no rule sets registered.
    pub fn empty() -> Self {
        Self {
            regulations: HashMap::new(),
            reports: VecDeque::new(),
            external_apis: HashMap::new(),
        }
    }

    /// Frameworks this checker can assess.
    pub fn registered_frameworks(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.regulations.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Register an external compliance API.
    ///
    /// Registration only records the endpoint; this crate performs no network
    /// I/O, so an external API never contributes to a verdict.
    pub fn register_external_api(&mut self, api: ExternalComplianceAPI) {
        self.external_apis.insert(api.name.clone(), api);
    }

    /// Registered external API names.
    pub fn external_api_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.external_apis.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of reports produced.
    pub fn report_count(&self) -> usize {
        self.reports.len()
    }

    /// Assess one framework over the events in scope.
    fn assess(
        checker: &RegulationChecker,
        events: &[&AuditEvent],
    ) -> (ComplianceAssessment, ComplianceStatus) {
        let mut findings = Vec::new();
        let mut recommendations: Vec<String> = Vec::new();
        let mut checks = 0usize;
        let mut passes = 0usize;
        let mut weighted_violation = 0.0f64;
        let mut weight_total = 0.0f64;
        let mut worst: Option<RuleSeverity> = None;
        let mut violations_by_severity: HashMap<&'static str, (f64, usize)> = HashMap::new();

        for event in events {
            for rule in &checker.rules {
                let result = (rule.evaluation_fn)(event);
                checks += 1;
                let weight = severity_weight(rule.severity);
                weight_total += weight;
                if result.passed {
                    passes += 1;
                } else {
                    weighted_violation += weight;
                    let label = severity_label(rule.severity);
                    let entry = violations_by_severity.entry(label).or_insert((weight, 0));
                    entry.1 += 1;
                    worst = Some(match worst {
                        Some(current) if severity_weight(current) >= weight => current,
                        _ => rule.severity,
                    });
                    findings.push(ComplianceFinding {
                        id: format!("{}::{}", rule.id, event.id),
                        rule: rule.name.clone(),
                        status: ComplianceStatus::NonCompliant(result.message.clone()),
                        evidence: vec![
                            format!("event_id={}", event.id),
                            format!(
                                "event_type={}",
                                super::hashing::event_type_key(&event.event_type)
                            ),
                            format!("timestamp={}", event.timestamp),
                        ],
                        impact: severity_impact(rule.severity),
                    });
                    for recommendation in result.recommendations {
                        if !recommendations.contains(&recommendation) {
                            recommendations.push(recommendation);
                        }
                    }
                }
            }
        }

        let compliance_score = if checks == 0 {
            0.0
        } else {
            passes as f64 / checks as f64
        };
        let risk_score = if weight_total == 0.0 {
            0.0
        } else {
            (weighted_violation / weight_total).clamp(0.0, 1.0)
        };
        let risk_factors: Vec<RiskFactor> = {
            let mut factors: Vec<RiskFactor> = violations_by_severity
                .into_iter()
                .map(|(label, (weight, count))| RiskFactor {
                    name: format!("{label}_violations"),
                    weight,
                    value: count as f64,
                    description: format!("{count} {label}-severity rule violation(s)"),
                })
                .collect();
            factors.sort_by(|left, right| left.name.cmp(&right.name));
            factors
        };

        let status = if checks == 0 {
            ComplianceStatus::RequiresReview(format!(
                "no audit events fall inside the reporting period, so {} compliance cannot be \
                 asserted",
                framework_key(&checker.framework)
            ))
        } else {
            match worst {
                None => ComplianceStatus::Compliant,
                Some(RuleSeverity::Critical) | Some(RuleSeverity::High) => {
                    ComplianceStatus::NonCompliant(format!(
                        "{} of {checks} checks failed, including at least one high-severity rule",
                        checks - passes
                    ))
                }
                Some(_) => ComplianceStatus::RequiresReview(format!(
                    "{} of {checks} checks failed at medium severity or below",
                    checks - passes
                )),
            }
        };

        let assessment = ComplianceAssessment {
            framework: checker.framework.clone(),
            compliance_score,
            findings,
            recommendations,
            risk_assessment: RiskAssessment {
                risk_score,
                risk_factors,
                mitigations: Vec::new(),
                // No remediation has been executed at report time, so the
                // residual risk equals the assessed risk. Reporting a lower
                // residual would be unearned credit.
                residual_risk: risk_score,
            },
        };
        (assessment, status)
    }

    /// Generate a compliance report from a real audit trail.
    pub fn generate_report(
        &self,
        frameworks: &[ComplianceFramework],
        period: ReportingPeriod,
        audit_trail: &AuditTrail,
    ) -> Result<ComplianceReport> {
        let (window_start, window_end) = period_bounds(&period)?;
        let in_scope: Vec<&AuditEvent> = audit_trail
            .events()
            .filter(|event| event.timestamp >= window_start && event.timestamp <= window_end)
            .collect();

        let mut assessments = HashMap::new();
        let mut statuses = Vec::new();
        let mut summary_lines = Vec::new();

        if frameworks.is_empty() {
            statuses.push(ComplianceStatus::RequiresReview(
                "no compliance framework was requested".to_string(),
            ));
            summary_lines.push("No framework was requested; nothing was assessed.".to_string());
        }

        for framework in frameworks {
            let key = framework_key(framework);
            match self.regulations.get(&key) {
                Some(checker) => {
                    let (assessment, status) = Self::assess(checker, &in_scope);
                    summary_lines.push(format!(
                        "{key}: {} finding(s), pass rate {:.3}, risk {:.3}.",
                        assessment.findings.len(),
                        assessment.compliance_score,
                        assessment.risk_assessment.risk_score
                    ));
                    assessments.insert(framework.clone(), assessment);
                    statuses.push(status);
                }
                None => {
                    summary_lines.push(format!(
                        "{key}: no rule set is implemented in this build; the framework was not \
                         assessed."
                    ));
                    statuses.push(ComplianceStatus::RequiresReview(format!(
                        "no rule set is implemented for {key}, so its compliance was not assessed"
                    )));
                }
            }
        }

        let overall_status = combine_statuses(&statuses);
        let report = ComplianceReport {
            id: format!("report_{}", self.reports.len()),
            timestamp: unix_timestamp()?,
            period,
            frameworks: frameworks.to_vec(),
            overall_status,
            assessments,
            executive_summary: format!(
                "{} event(s) in scope out of {} recorded. {}",
                in_scope.len(),
                audit_trail.len(),
                summary_lines.join(" ")
            ),
            format: ReportFormat::JSON,
        };
        Ok(report)
    }
}

impl Default for RegulatoryComplianceChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// A short label for a severity.
fn severity_label(severity: RuleSeverity) -> &'static str {
    match severity {
        RuleSeverity::Critical => "critical",
        RuleSeverity::High => "high",
        RuleSeverity::Medium => "medium",
        RuleSeverity::Low => "low",
        RuleSeverity::Info => "info",
    }
}

/// Inclusive `[start, end]` timestamp bounds of a reporting period, ending now.
fn period_bounds(period: &ReportingPeriod) -> Result<(u64, u64)> {
    const DAY: u64 = 86_400;
    let now = unix_timestamp()?;
    let span = match period {
        ReportingPeriod::Daily => DAY,
        ReportingPeriod::Weekly => 7 * DAY,
        ReportingPeriod::Monthly => 30 * DAY,
        ReportingPeriod::Quarterly => 91 * DAY,
        ReportingPeriod::Annual => 365 * DAY,
        ReportingPeriod::Custom(start, end) => {
            let (start, end) = if start <= end {
                (*start, *end)
            } else {
                (*end, *start)
            };
            return Ok((start, end));
        }
    };
    Ok((now.saturating_sub(span), now))
}

/// Combine per-framework statuses into the overall verdict.
///
/// `Compliant` requires *every* framework to be compliant; a single
/// `RequiresReview` (which is what an unassessed framework yields) is enough to
/// withhold the verdict.
fn combine_statuses(statuses: &[ComplianceStatus]) -> ComplianceStatus {
    let mut non_compliant = Vec::new();
    let mut review = Vec::new();
    for status in statuses {
        match status {
            ComplianceStatus::NonCompliant(reason) => non_compliant.push(reason.clone()),
            ComplianceStatus::RequiresReview(reason) => review.push(reason.clone()),
            ComplianceStatus::Compliant | ComplianceStatus::NotApplicable => {}
        }
    }
    if !non_compliant.is_empty() {
        ComplianceStatus::NonCompliant(non_compliant.join("; "))
    } else if !review.is_empty() {
        ComplianceStatus::RequiresReview(review.join("; "))
    } else if statuses.is_empty() {
        ComplianceStatus::RequiresReview("nothing was assessed".to_string())
    } else {
        ComplianceStatus::Compliant
    }
}

/// Live compliance monitoring over the event stream.
pub struct ComplianceMonitor {
    /// Frameworks being monitored.
    frameworks: Vec<ComplianceFramework>,
    /// Rules per framework key.
    rules: HashMap<String, Vec<ComplianceRule>>,
    /// Violations observed, most recent last.
    violations: VecDeque<ComplianceViolation>,
    /// Remediation actions, keyed by rule id.
    remediation_actions: HashMap<String, RemediationAction>,
    /// Cap on retained violations.
    violation_capacity: usize,
}

impl ComplianceMonitor {
    /// Monitor every framework this build can assess.
    pub fn new() -> Self {
        Self::for_frameworks(&supported_frameworks())
    }

    /// Monitor a specific set of frameworks.
    ///
    /// Frameworks without an implemented rule set contribute no checks; they
    /// are reported by [`ComplianceMonitor::unassessed_frameworks`] so a caller
    /// can tell "monitored and clean" from "not monitored at all".
    pub fn for_frameworks(frameworks: &[ComplianceFramework]) -> Self {
        let mut rules = HashMap::new();
        for framework in frameworks {
            if let Some(framework_rules) = rules_for(framework) {
                rules.insert(framework_key(framework), framework_rules);
            }
        }
        Self {
            frameworks: frameworks.to_vec(),
            rules,
            violations: VecDeque::new(),
            remediation_actions: HashMap::new(),
            violation_capacity: 10_000,
        }
    }

    /// Frameworks that were requested but have no rule set in this build.
    pub fn unassessed_frameworks(&self) -> Vec<String> {
        self.frameworks
            .iter()
            .map(framework_key)
            .filter(|key| !self.rules.contains_key(key))
            .collect()
    }

    /// Number of rules actually evaluated per event.
    pub fn active_rule_count(&self) -> usize {
        self.rules.values().map(|rules| rules.len()).sum()
    }

    /// Register a remediation action for a rule id.
    pub fn register_remediation_action(&mut self, rule_id: String, action: RemediationAction) {
        self.remediation_actions.insert(rule_id, action);
    }

    /// Recorded violations, oldest first.
    pub fn violations(&self) -> impl Iterator<Item = &ComplianceViolation> {
        self.violations.iter()
    }

    /// Number of recorded violations.
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }

    /// The remediation action registered for a rule, if any.
    pub fn remediation_action(&self, rule_id: &str) -> Option<&RemediationAction> {
        self.remediation_actions.get(rule_id)
    }

    /// Evaluate every active rule against `event`, recording violations.
    ///
    /// Returns the number of violations recorded for this event.
    pub fn check_event(&mut self, event: &AuditEvent) -> Result<usize> {
        let mut pending = Vec::new();
        for framework in &self.frameworks {
            let key = framework_key(framework);
            let Some(rules) = self.rules.get(&key) else {
                continue;
            };
            for rule in rules {
                let result = (rule.evaluation_fn)(event);
                if !result.passed {
                    pending.push((framework.clone(), rule.id.clone(), rule.severity, result));
                }
            }
        }

        let recorded = pending.len();
        for (framework, rule_id, severity, result) in pending {
            self.record_violation(&framework, &rule_id, severity, event, result)?;
        }
        Ok(recorded)
    }

    /// Record a violation.
    fn record_violation(
        &mut self,
        framework: &ComplianceFramework,
        rule_id: &str,
        severity: RuleSeverity,
        event: &AuditEvent,
        result: ComplianceRuleResult,
    ) -> Result<()> {
        let violation = ComplianceViolation {
            id: format!("violation_{}", self.violations.len()),
            timestamp: unix_timestamp()?,
            rule_id: rule_id.to_string(),
            severity,
            framework: framework.clone(),
            description: result.message,
            remediation_status: RemediationStatus::Open,
            audit_event_id: event.id.clone(),
        };
        self.violations.push_back(violation);
        while self.violations.len() > self.violation_capacity {
            let _ = self.violations.pop_front();
        }
        Ok(())
    }
}

impl Default for ComplianceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::enhanced_audit::types::{AuditEventData, PrivacyContext};

    fn compliant_event(id: &str) -> AuditEvent {
        AuditEvent {
            id: id.to_string(),
            timestamp: unix_timestamp().unwrap_or_default(),
            event_type: AuditEventType::DataAccess,
            actor: "trainer".to_string(),
            data: AuditEventData {
                description: "read a training shard".to_string(),
                affected_data_subjects: vec!["subject-a".to_string()],
                data_categories: vec!["personal_data".to_string()],
                processing_purposes: vec!["ml_training".to_string()],
                legal_basis: vec!["consent".to_string()],
                technical_measures: vec!["differential_privacy".to_string()],
                metadata: HashMap::new(),
            },
            privacy_context: PrivacyContext {
                epsilon_budget: 0.5,
                delta_budget: 1e-6,
                privacy_mechanism: "dp_sgd".to_string(),
                data_minimization: true,
                purpose_limitation: true,
                storage_limitation: true,
            },
            signature: Some(vec![1u8; 32]),
            compliance_annotations: HashMap::new(),
        }
    }

    fn trail_with(events: Vec<AuditEvent>) -> AuditTrail {
        let mut trail = AuditTrail::new();
        for event in events {
            let ok = trail.add_event(event);
            assert!(ok.is_ok());
        }
        trail
    }

    #[test]
    fn an_empty_trail_never_reports_compliant() {
        // Regression: `generate_report` used to return a hardcoded
        // `ComplianceStatus::Compliant` with empty assessments for any input,
        // including an audit trail with no events at all.
        let checker = RegulatoryComplianceChecker::new();
        let trail = AuditTrail::new();
        let report = match checker.generate_report(
            &[ComplianceFramework::GDPR],
            ReportingPeriod::Daily,
            &trail,
        ) {
            Ok(report) => report,
            Err(err) => panic!("report failed: {err}"),
        };
        match report.overall_status {
            ComplianceStatus::RequiresReview(ref reason) => {
                assert!(reason.contains("no audit events"), "got: {reason}");
            }
            other => panic!("an empty trail must require review, got {other:?}"),
        }
    }

    #[test]
    fn a_violating_event_makes_the_report_non_compliant() {
        let checker = RegulatoryComplianceChecker::new();
        let mut event = compliant_event("bad");
        event.data.legal_basis.clear();
        let trail = trail_with(vec![event]);

        let report = match checker.generate_report(
            &[ComplianceFramework::GDPR],
            ReportingPeriod::Daily,
            &trail,
        ) {
            Ok(report) => report,
            Err(err) => panic!("report failed: {err}"),
        };
        match report.overall_status {
            ComplianceStatus::NonCompliant(_) => {}
            other => panic!("a missing legal basis must be non-compliant, got {other:?}"),
        }
        let assessment = match report.assessments.get(&ComplianceFramework::GDPR) {
            Some(assessment) => assessment,
            None => panic!("the GDPR assessment must be present"),
        };
        assert!(!assessment.findings.is_empty());
        assert!(assessment.compliance_score < 1.0);
        assert!(assessment.risk_assessment.risk_score > 0.0);
        assert!(!assessment.recommendations.is_empty());
    }

    #[test]
    fn a_clean_event_can_reach_compliant() {
        let checker = RegulatoryComplianceChecker::new();
        let trail = trail_with(vec![compliant_event("good")]);
        let report = match checker.generate_report(
            &[ComplianceFramework::GDPR, ComplianceFramework::HIPAA],
            ReportingPeriod::Daily,
            &trail,
        ) {
            Ok(report) => report,
            Err(err) => panic!("report failed: {err}"),
        };
        match report.overall_status {
            ComplianceStatus::Compliant => {}
            other => panic!("a clean event must be compliant, got {other:?}"),
        }
        let gdpr = match report.assessments.get(&ComplianceFramework::GDPR) {
            Some(assessment) => assessment,
            None => panic!("missing GDPR assessment"),
        };
        assert!((gdpr.compliance_score - 1.0).abs() < 1e-12);
        assert_eq!(gdpr.risk_assessment.risk_score, 0.0);
    }

    #[test]
    fn a_framework_with_no_rule_set_is_never_reported_compliant() {
        let checker = RegulatoryComplianceChecker::new();
        let trail = trail_with(vec![compliant_event("good")]);
        let report = match checker.generate_report(
            &[ComplianceFramework::SOX],
            ReportingPeriod::Daily,
            &trail,
        ) {
            Ok(report) => report,
            Err(err) => panic!("report failed: {err}"),
        };
        match report.overall_status {
            ComplianceStatus::RequiresReview(ref reason) => {
                assert!(reason.contains("no rule set"), "got: {reason}");
            }
            other => panic!("an unimplemented framework must require review, got {other:?}"),
        }
        assert!(report.assessments.is_empty());
    }

    #[test]
    fn requesting_no_framework_is_not_compliance() {
        let checker = RegulatoryComplianceChecker::new();
        let trail = trail_with(vec![compliant_event("good")]);
        let report = match checker.generate_report(&[], ReportingPeriod::Daily, &trail) {
            Ok(report) => report,
            Err(err) => panic!("report failed: {err}"),
        };
        assert!(matches!(
            report.overall_status,
            ComplianceStatus::RequiresReview(_)
        ));
    }

    #[test]
    fn events_outside_the_reporting_period_are_excluded() {
        let checker = RegulatoryComplianceChecker::new();
        let mut old = compliant_event("old");
        old.timestamp = 1_000;
        let trail = trail_with(vec![old]);
        let report = match checker.generate_report(
            &[ComplianceFramework::GDPR],
            ReportingPeriod::Daily,
            &trail,
        ) {
            Ok(report) => report,
            Err(err) => panic!("report failed: {err}"),
        };
        assert!(matches!(
            report.overall_status,
            ComplianceStatus::RequiresReview(_)
        ));
        assert!(report.executive_summary.contains("0 event(s) in scope"));
    }

    #[test]
    fn a_custom_period_selects_exactly_the_events_inside_it() {
        let checker = RegulatoryComplianceChecker::new();
        let mut first = compliant_event("first");
        first.timestamp = 100;
        let mut second = compliant_event("second");
        second.timestamp = 200;
        let trail = trail_with(vec![first, second]);

        let report = match checker.generate_report(
            &[ComplianceFramework::GDPR],
            ReportingPeriod::Custom(150, 250),
            &trail,
        ) {
            Ok(report) => report,
            Err(err) => panic!("report failed: {err}"),
        };
        assert!(report.executive_summary.contains("1 event(s) in scope"));
    }

    #[test]
    fn the_monitor_records_violations_instead_of_printing_them() {
        // Regression: the monitor's only reaction to a failing rule was
        // `eprintln!`, and `record_violation` was dead code.
        let mut monitor = ComplianceMonitor::new();
        assert!(monitor.active_rule_count() > 0);

        match monitor.check_event(&compliant_event("good")) {
            Ok(0) => {}
            Ok(count) => panic!("a clean event recorded {count} violations"),
            Err(err) => panic!("check_event failed: {err}"),
        }
        assert_eq!(monitor.violation_count(), 0);

        let mut bad = compliant_event("bad");
        bad.privacy_context.data_minimization = false;
        bad.data.technical_measures.clear();
        let recorded = match monitor.check_event(&bad) {
            Ok(count) => count,
            Err(err) => panic!("check_event failed: {err}"),
        };
        assert!(recorded >= 2, "recorded {recorded} violations");
        assert_eq!(monitor.violation_count(), recorded);
        let first = match monitor.violations().next() {
            Some(violation) => violation,
            None => panic!("a violation must be retained"),
        };
        assert_eq!(first.audit_event_id, "bad");
        assert!(matches!(first.remediation_status, RemediationStatus::Open));
    }

    #[test]
    fn a_monitor_reports_which_requested_frameworks_it_cannot_assess() {
        let monitor = ComplianceMonitor::for_frameworks(&[
            ComplianceFramework::GDPR,
            ComplianceFramework::ISO27001,
        ]);
        assert_eq!(
            monitor.unassessed_frameworks(),
            vec!["ISO27001".to_string()]
        );
    }

    #[test]
    fn hipaa_rejects_a_data_release_without_a_de_identification_measure() {
        let mut monitor = ComplianceMonitor::for_frameworks(&[ComplianceFramework::HIPAA]);
        let mut event = compliant_event("raw-release");
        event.data.technical_measures = vec!["tls".to_string()];
        let recorded = match monitor.check_event(&event) {
            Ok(count) => count,
            Err(err) => panic!("check_event failed: {err}"),
        };
        assert_eq!(recorded, 1);
        let violation = match monitor.violations().next() {
            Some(violation) => violation,
            None => panic!("expected a violation"),
        };
        assert_eq!(violation.rule_id, "hipaa-164-514-b-de-identification");
    }

    #[test]
    fn ccpa_only_requires_data_subjects_on_deletion_events() {
        let mut monitor = ComplianceMonitor::for_frameworks(&[ComplianceFramework::CCPA]);
        let mut deletion = compliant_event("deletion");
        deletion.event_type = AuditEventType::DataDeletion;
        deletion.data.affected_data_subjects.clear();
        let recorded = match monitor.check_event(&deletion) {
            Ok(count) => count,
            Err(err) => panic!("check_event failed: {err}"),
        };
        assert_eq!(recorded, 1);

        let mut config_change = compliant_event("config");
        config_change.event_type = AuditEventType::ConfigurationChange;
        config_change.data.affected_data_subjects.clear();
        config_change.data.data_categories.clear();
        let recorded = match monitor.check_event(&config_change) {
            Ok(count) => count,
            Err(err) => panic!("check_event failed: {err}"),
        };
        assert_eq!(
            recorded, 0,
            "a configuration change touches no consumer data"
        );
    }
}
