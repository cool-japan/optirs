// Academic conference integration and submission tools
//
// This module provides tools for managing conference submissions,
// tracking deadlines, and preparing submission materials.

use crate::error::{OptimError, Result};
use chrono::{DateTime, Datelike, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Conference database and submission manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConferenceManager {
    /// Known conferences
    pub conferences: HashMap<String, Conference>,
    /// Submission tracking
    pub submissions: Vec<Submission>,
    /// Deadline alerts
    pub alerts: Vec<DeadlineAlert>,
}

/// Academic conference information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conference {
    /// Conference identifier
    pub id: String,
    /// Conference name
    pub name: String,
    /// Conference abbreviation
    pub abbreviation: String,
    /// Conference description
    pub description: String,
    /// Conference URL
    pub url: String,
    /// Conference ranking/tier
    pub ranking: ConferenceRanking,
    /// Research areas
    pub research_areas: Vec<String>,
    /// Annual occurrence
    pub annual: bool,
    /// Conference series information
    pub series_info: SeriesInfo,
    /// Important dates
    pub dates: ConferenceDates,
    /// Submission requirements
    pub requirements: SubmissionRequirements,
    /// Review process
    pub review_process: ReviewProcess,
}

/// Conference ranking/tier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConferenceRanking {
    /// Top-tier (A*)
    TopTier,
    /// High-quality (A)
    HighQuality,
    /// Good (B)
    Good,
    /// Acceptable (C)
    Acceptable,
    /// Emerging
    Emerging,
    /// Workshop
    Workshop,
    /// Unranked
    Unranked,
}

/// Conference series information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesInfo {
    /// Series number (e.g., 35th)
    pub series_number: u32,
    /// Year
    pub year: u32,
    /// Location
    pub location: String,
    /// Country
    pub country: String,
    /// Conference format
    pub format: ConferenceFormat,
}

/// Conference format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConferenceFormat {
    /// In-person conference
    InPerson,
    /// Virtual conference
    Virtual,
    /// Hybrid conference
    Hybrid,
}

/// Important conference dates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConferenceDates {
    /// Abstract submission deadline
    pub abstract_deadline: Option<DateTime<Utc>>,
    /// Paper submission deadline
    pub paper_deadline: DateTime<Utc>,
    /// Notification date
    pub notification_date: DateTime<Utc>,
    /// Camera-ready deadline
    pub camera_ready_deadline: DateTime<Utc>,
    /// Conference start date
    pub conference_start: DateTime<Utc>,
    /// Conference end date
    pub conference_end: DateTime<Utc>,
}

/// Submission requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmissionRequirements {
    /// Page limit
    pub page_limit: u32,
    /// Word limit
    pub word_limit: Option<u32>,
    /// Format requirements
    pub format: FormatRequirements,
    /// Required sections
    pub required_sections: Vec<String>,
    /// Supplementary material allowed
    pub supplementary_allowed: bool,
    /// Anonymous submission required
    pub anonymous_submission: bool,
    /// Double-blind review
    pub double_blind: bool,
}

/// Format requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatRequirements {
    /// Document template
    pub template: String,
    /// Font size
    pub font_size: u32,
    /// Line spacing
    pub line_spacing: f64,
    /// Margins
    pub margins: String,
    /// Citation style
    pub citation_style: String,
    /// File format
    pub file_format: Vec<String>,
}

/// Review process information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewProcess {
    /// Number of reviewers per paper
    pub reviewers_per_paper: u32,
    /// Review criteria
    pub review_criteria: Vec<String>,
    /// Rebuttal allowed
    pub rebuttal_allowed: bool,
    /// Acceptance rate (if known)
    pub acceptance_rate: Option<f64>,
    /// Review format
    pub review_format: ReviewFormat,
}

/// Review format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReviewFormat {
    /// Numerical scores
    NumericalScores,
    /// Written reviews only
    WrittenOnly,
    /// Mixed format
    Mixed,
}

/// Conference submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Submission {
    /// Submission ID
    pub id: String,
    /// Conference ID
    pub conference_id: String,
    /// Paper/publication ID
    pub paper_id: String,
    /// Submission status
    pub status: SubmissionStatus,
    /// Submission date
    pub submitted_at: DateTime<Utc>,
    /// Track/category
    pub track: Option<String>,
    /// Submission materials
    pub materials: SubmissionMaterials,
    /// Review information
    pub reviews: Vec<Review>,
    /// Decision information
    pub decision: Option<Decision>,
}

/// Submission status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SubmissionStatus {
    /// Draft
    Draft,
    /// Submitted
    Submitted,
    /// Under review
    UnderReview,
    /// Rebuttal period
    Rebuttal,
    /// Decision made
    Decided,
    /// Camera-ready submitted
    CameraReady,
    /// Withdrawn
    Withdrawn,
}

/// Submission materials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmissionMaterials {
    /// Main paper file
    pub paper_file: String,
    /// Supplementary materials
    pub supplementary_files: Vec<String>,
    /// Abstract
    pub abstracttext: String,
    /// Keywords
    pub keywords: Vec<String>,
    /// Author information (if not anonymous)
    pub authors: Option<Vec<String>>,
}

/// Review information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Review {
    /// Review ID
    pub id: String,
    /// Reviewer (anonymous)
    pub reviewer: String,
    /// Overall score
    pub overall_score: Option<f64>,
    /// Detailed scores
    pub detailed_scores: HashMap<String, f64>,
    /// Written review
    pub reviewtext: String,
    /// Recommendation
    pub recommendation: ReviewRecommendation,
    /// Review date
    pub reviewed_at: DateTime<Utc>,
}

/// Review recommendations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReviewRecommendation {
    /// Strong accept
    StrongAccept,
    /// Accept
    Accept,
    /// Weak accept
    WeakAccept,
    /// Borderline
    Borderline,
    /// Weak reject
    WeakReject,
    /// Reject
    Reject,
    /// Strong reject
    StrongReject,
}

/// Conference decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    /// Decision outcome
    pub outcome: DecisionOutcome,
    /// Decision date
    pub decided_at: DateTime<Utc>,
    /// Editor comments
    pub editor_comments: Option<String>,
    /// Required revisions
    pub required_revisions: Vec<String>,
}

/// Decision outcomes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecisionOutcome {
    /// Accept
    Accept,
    /// Accept with minor revisions
    AcceptMinorRevisions,
    /// Accept with major revisions
    AcceptMajorRevisions,
    /// Conditional accept
    ConditionalAccept,
    /// Reject
    Reject,
    /// Reject and resubmit
    RejectAndResubmit,
}

/// Deadline alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlineAlert {
    /// Alert ID
    pub id: String,
    /// Conference ID
    pub conference_id: String,
    /// Deadline type
    pub deadline_type: DeadlineType,
    /// Alert date
    pub alert_date: DateTime<Utc>,
    /// Days before deadline
    pub days_before: u32,
    /// Alert message
    pub message: String,
    /// Alert sent
    pub sent: bool,
}

/// Types of deadlines
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeadlineType {
    /// Abstract submission
    AbstractSubmission,
    /// Paper submission
    PaperSubmission,
    /// Notification
    Notification,
    /// Camera-ready
    CameraReady,
    /// Conference start
    ConferenceStart,
}

impl Default for ConferenceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ConferenceManager {
    /// Create a new conference manager
    pub fn new() -> Self {
        Self {
            conferences: HashMap::new(),
            submissions: Vec::new(),
            alerts: Vec::new(),
        }
    }

    /// Add a conference to the database
    pub fn add_conference(&mut self, conference: Conference) {
        self.conferences.insert(conference.id.clone(), conference);
    }

    /// Submit a paper to a conference
    pub fn submit_paper(
        &mut self,
        conference_id: &str,
        paper_id: &str,
        materials: SubmissionMaterials,
    ) -> Result<String> {
        if !self.conferences.contains_key(conference_id) {
            return Err(OptimError::InvalidConfig(format!(
                "Conference '{}' not found",
                conference_id
            )));
        }

        let submission_id = uuid::Uuid::new_v4().to_string();
        let submission = Submission {
            id: submission_id.clone(),
            conference_id: conference_id.to_string(),
            paper_id: paper_id.to_string(),
            status: SubmissionStatus::Submitted,
            submitted_at: Utc::now(),
            track: None,
            materials,
            reviews: Vec::new(),
            decision: None,
        };

        self.submissions.push(submission);
        Ok(submission_id)
    }

    /// Get upcoming deadlines
    pub fn get_upcoming_deadlines(
        &self,
        days_ahead: u32,
    ) -> Vec<(&Conference, DeadlineType, DateTime<Utc>)> {
        let mut deadlines = Vec::new();
        let now = Utc::now();
        let future_limit = now + chrono::Duration::days(days_ahead as i64);

        for conference in self.conferences.values() {
            let dates = &conference.dates;

            if let Some(abstract_deadline) = dates.abstract_deadline {
                if abstract_deadline > now && abstract_deadline <= future_limit {
                    deadlines.push((
                        conference,
                        DeadlineType::AbstractSubmission,
                        abstract_deadline,
                    ));
                }
            }

            if dates.paper_deadline > now && dates.paper_deadline <= future_limit {
                deadlines.push((
                    conference,
                    DeadlineType::PaperSubmission,
                    dates.paper_deadline,
                ));
            }

            if dates.notification_date > now && dates.notification_date <= future_limit {
                deadlines.push((
                    conference,
                    DeadlineType::Notification,
                    dates.notification_date,
                ));
            }

            if dates.camera_ready_deadline > now && dates.camera_ready_deadline <= future_limit {
                deadlines.push((
                    conference,
                    DeadlineType::CameraReady,
                    dates.camera_ready_deadline,
                ));
            }

            if dates.conference_start > now && dates.conference_start <= future_limit {
                deadlines.push((
                    conference,
                    DeadlineType::ConferenceStart,
                    dates.conference_start,
                ));
            }
        }

        // Sort by deadline date
        deadlines.sort_by_key(|a| a.2);
        deadlines
    }

    /// Scan all known conferences and create [`DeadlineAlert`]s for any
    /// still-future deadline that falls within one of the `days_before`
    /// thresholds (e.g. `&[30, 7, 1]` for month/week/day-out reminders),
    /// appending them to `self.alerts`. Idempotent: calling this repeatedly
    /// (e.g. once a day) will not create duplicate alerts for the same
    /// (conference, deadline type, threshold) combination. Returns the
    /// alerts newly created by this call.
    pub fn generate_deadline_alerts(&mut self, days_before: &[u32]) -> Vec<DeadlineAlert> {
        let now = Utc::now();

        // Snapshot deadlines up front so we are not borrowing
        // `self.conferences` while mutating `self.alerts` below.
        let mut deadline_entries: Vec<(String, DeadlineType, DateTime<Utc>)> = Vec::new();
        for conference in self.conferences.values() {
            let dates = &conference.dates;
            if let Some(abstract_deadline) = dates.abstract_deadline {
                deadline_entries.push((
                    conference.id.clone(),
                    DeadlineType::AbstractSubmission,
                    abstract_deadline,
                ));
            }
            deadline_entries.push((
                conference.id.clone(),
                DeadlineType::PaperSubmission,
                dates.paper_deadline,
            ));
            deadline_entries.push((
                conference.id.clone(),
                DeadlineType::Notification,
                dates.notification_date,
            ));
            deadline_entries.push((
                conference.id.clone(),
                DeadlineType::CameraReady,
                dates.camera_ready_deadline,
            ));
            deadline_entries.push((
                conference.id.clone(),
                DeadlineType::ConferenceStart,
                dates.conference_start,
            ));
        }

        let mut new_alerts = Vec::new();
        for (conference_id, deadline_type, deadline) in deadline_entries {
            if deadline <= now {
                continue;
            }
            let days_remaining = (deadline - now).num_days().max(0) as u32;

            for &threshold in days_before {
                if days_remaining > threshold {
                    continue;
                }

                let already_exists = self.alerts.iter().any(|alert| {
                    alert.conference_id == conference_id
                        && alert.deadline_type == deadline_type
                        && alert.days_before == threshold
                });
                if already_exists {
                    continue;
                }

                let alert = DeadlineAlert {
                    id: uuid::Uuid::new_v4().to_string(),
                    conference_id: conference_id.clone(),
                    deadline_type: deadline_type.clone(),
                    alert_date: now,
                    days_before: threshold,
                    message: format!(
                        "{deadline_type:?} deadline for conference '{conference_id}' is in \
                         {days_remaining} day(s) ({deadline})"
                    ),
                    sent: false,
                };
                self.alerts.push(alert.clone());
                new_alerts.push(alert);
            }
        }

        new_alerts
    }

    /// Alerts that have not yet been marked as sent, most recent first.
    pub fn pending_alerts(&self) -> Vec<&DeadlineAlert> {
        self.alerts.iter().filter(|alert| !alert.sent).collect()
    }

    /// Mark an alert as sent (e.g. after successfully notifying a user).
    pub fn mark_alert_sent(&mut self, alert_id: &str) -> Result<()> {
        let alert = self
            .alerts
            .iter_mut()
            .find(|alert| alert.id == alert_id)
            .ok_or_else(|| OptimError::InvalidConfig(format!("alert '{alert_id}' not found")))?;
        alert.sent = true;
        Ok(())
    }

    /// Search conferences by research area
    pub fn search_conferences(&self, research_area: &str) -> Vec<&Conference> {
        self.conferences
            .values()
            .filter(|conf| {
                conf.research_areas
                    .iter()
                    .any(|area| area.to_lowercase().contains(&research_area.to_lowercase()))
            })
            .collect()
    }

    /// Get conferences by ranking
    pub fn get_conferences_by_ranking(&self, ranking: ConferenceRanking) -> Vec<&Conference> {
        self.conferences
            .values()
            .filter(|conf| conf.ranking == ranking)
            .collect()
    }

    /// Create standard ML/AI conferences, using each conference's next
    /// upcoming edition (by real submission deadline) rather than a fixed
    /// calendar year, so [`Self::get_upcoming_deadlines`] can actually find
    /// them no matter when this is called.
    pub fn load_standard_conferences(&mut self) {
        self.add_conference(Self::next_upcoming_edition(Self::create_neurips_conference));
        self.add_conference(Self::next_upcoming_edition(Self::create_icml_conference));
        self.add_conference(Self::next_upcoming_edition(Self::create_iclr_conference));
        self.add_conference(Self::next_upcoming_edition(Self::create_aaai_conference));
        self.add_conference(Self::next_upcoming_edition(Self::create_ijcai_conference));
    }

    /// Build successive editions of a conference (via `build`, which takes
    /// the edition's label year, e.g. 2027 for "NeurIPS 2027") until one is
    /// found whose paper submission deadline has not yet passed, and return
    /// that edition.
    ///
    /// Conference templates below are seasonal (same month/day pattern every
    /// year), so trying a handful of consecutive label years is always
    /// sufficient; this makes the loaded deadlines self-correcting as real
    /// time passes, instead of frozen at whatever year they were written in.
    fn next_upcoming_edition(build: fn(i32) -> Conference) -> Conference {
        let now = Utc::now();
        let start_year = now.year();
        let last_candidate = start_year + 3;
        for candidate_year in start_year..=last_candidate {
            let conference = build(candidate_year);
            if conference.dates.paper_deadline > now {
                return conference;
            }
        }
        // Unreachable in practice for an annual conference (deadlines cannot
        // trail 3+ years behind "now" for every candidate), but stay honest
        // and return a well-formed, self-consistent edition rather than
        // panicking.
        build(last_candidate)
    }

    fn create_neurips_conference(edition_year: i32) -> Conference {
        Conference {
            id: format!("neurips{edition_year}"),
            name: "Conference on Neural Information Processing Systems".to_string(),
            abbreviation: "NeurIPS".to_string(),
            description: "Premier conference on neural information processing systems".to_string(),
            url: "https://neurips.cc/".to_string(),
            ranking: ConferenceRanking::TopTier,
            research_areas: vec![
                "Machine Learning".to_string(),
                "Deep Learning".to_string(),
                "Neural Networks".to_string(),
                "Optimization".to_string(),
            ],
            annual: true,
            series_info: SeriesInfo {
                series_number: 38,
                year: edition_year as u32,
                location: "Vancouver".to_string(),
                country: "Canada".to_string(),
                format: ConferenceFormat::Hybrid,
            },
            dates: ConferenceDates {
                abstract_deadline: Some(
                    chrono::Utc
                        .with_ymd_and_hms(edition_year, 5, 15, 23, 59, 59)
                        .single()
                        .expect("invalid datetime"),
                ),
                paper_deadline: chrono::Utc
                    .with_ymd_and_hms(edition_year, 5, 22, 23, 59, 59)
                    .single()
                    .expect("invalid datetime"),
                notification_date: chrono::Utc
                    .with_ymd_and_hms(edition_year, 9, 25, 12, 0, 0)
                    .single()
                    .expect("invalid datetime"),
                camera_ready_deadline: chrono::Utc
                    .with_ymd_and_hms(edition_year, 10, 30, 23, 59, 59)
                    .single()
                    .expect("invalid datetime"),
                conference_start: chrono::Utc
                    .with_ymd_and_hms(edition_year, 12, 10, 9, 0, 0)
                    .single()
                    .expect("invalid datetime"),
                conference_end: chrono::Utc
                    .with_ymd_and_hms(edition_year, 12, 16, 18, 0, 0)
                    .single()
                    .expect("invalid datetime"),
            },
            requirements: SubmissionRequirements {
                page_limit: 9,
                word_limit: None,
                format: FormatRequirements {
                    template: format!("NeurIPS {edition_year} LaTeX template"),
                    font_size: 10,
                    line_spacing: 1.0,
                    margins: "1 inch".to_string(),
                    citation_style: "NeurIPS".to_string(),
                    file_format: vec!["PDF".to_string()],
                },
                required_sections: vec![
                    "Abstract".to_string(),
                    "Introduction".to_string(),
                    "Related Work".to_string(),
                    "Method".to_string(),
                    "Experiments".to_string(),
                    "Conclusion".to_string(),
                ],
                supplementary_allowed: true,
                anonymous_submission: true,
                double_blind: true,
            },
            review_process: ReviewProcess {
                reviewers_per_paper: 3,
                review_criteria: vec![
                    "Technical Quality".to_string(),
                    "Novelty".to_string(),
                    "Significance".to_string(),
                    "Clarity".to_string(),
                ],
                rebuttal_allowed: true,
                acceptance_rate: Some(0.26), // Approximately 26%
                review_format: ReviewFormat::Mixed,
            },
        }
    }

    fn create_icml_conference(edition_year: i32) -> Conference {
        Conference {
            id: format!("icml{edition_year}"),
            name: "International Conference on Machine Learning".to_string(),
            abbreviation: "ICML".to_string(),
            description: "Premier international conference on machine learning".to_string(),
            url: "https://icml.cc/".to_string(),
            ranking: ConferenceRanking::TopTier,
            research_areas: vec![
                "Machine Learning".to_string(),
                "Optimization".to_string(),
                "Statistical Learning".to_string(),
                "Deep Learning".to_string(),
            ],
            annual: true,
            series_info: SeriesInfo {
                series_number: 41,
                year: edition_year as u32,
                location: "Vienna".to_string(),
                country: "Austria".to_string(),
                format: ConferenceFormat::Hybrid,
            },
            dates: ConferenceDates {
                abstract_deadline: None,
                paper_deadline: chrono::Utc
                    .with_ymd_and_hms(edition_year, 2, 1, 23, 59, 59)
                    .single()
                    .expect("invalid datetime"),
                notification_date: chrono::Utc
                    .with_ymd_and_hms(edition_year, 5, 1, 12, 0, 0)
                    .single()
                    .expect("invalid datetime"),
                camera_ready_deadline: chrono::Utc
                    .with_ymd_and_hms(edition_year, 6, 1, 23, 59, 59)
                    .single()
                    .expect("invalid datetime"),
                conference_start: chrono::Utc
                    .with_ymd_and_hms(edition_year, 7, 21, 9, 0, 0)
                    .single()
                    .expect("invalid datetime"),
                conference_end: chrono::Utc
                    .with_ymd_and_hms(edition_year, 7, 27, 18, 0, 0)
                    .single()
                    .expect("invalid datetime"),
            },
            requirements: SubmissionRequirements {
                page_limit: 8,
                word_limit: None,
                format: FormatRequirements {
                    template: format!("ICML {edition_year} LaTeX template"),
                    font_size: 10,
                    line_spacing: 1.0,
                    margins: "1 inch".to_string(),
                    citation_style: "ICML".to_string(),
                    file_format: vec!["PDF".to_string()],
                },
                required_sections: vec![
                    "Abstract".to_string(),
                    "Introduction".to_string(),
                    "Methods".to_string(),
                    "Results".to_string(),
                    "Conclusion".to_string(),
                ],
                supplementary_allowed: true,
                anonymous_submission: true,
                double_blind: true,
            },
            review_process: ReviewProcess {
                reviewers_per_paper: 3,
                review_criteria: vec![
                    "Technical Quality".to_string(),
                    "Clarity".to_string(),
                    "Originality".to_string(),
                    "Significance".to_string(),
                ],
                rebuttal_allowed: true,
                acceptance_rate: Some(0.23), // Approximately 23%
                review_format: ReviewFormat::Mixed,
            },
        }
    }

    fn create_iclr_conference(edition_year: i32) -> Conference {
        let prior_year = edition_year - 1;
        Conference {
            id: format!("iclr{edition_year}"),
            name: "International Conference on Learning Representations".to_string(),
            abbreviation: "ICLR".to_string(),
            description: "Conference focused on learning representations".to_string(),
            url: "https://iclr.cc/".to_string(),
            ranking: ConferenceRanking::TopTier,
            research_areas: vec![
                "Deep Learning".to_string(),
                "Representation Learning".to_string(),
                "Neural Networks".to_string(),
                "Optimization".to_string(),
            ],
            annual: true,
            series_info: SeriesInfo {
                series_number: 12,
                year: edition_year as u32,
                location: "Vienna".to_string(),
                country: "Austria".to_string(),
                format: ConferenceFormat::Hybrid,
            },
            dates: ConferenceDates {
                abstract_deadline: Some(
                    chrono::Utc
                        .with_ymd_and_hms(prior_year, 9, 28, 23, 59, 59)
                        .single()
                        .expect("invalid datetime"),
                ),
                paper_deadline: chrono::Utc
                    .with_ymd_and_hms(prior_year, 10, 2, 23, 59, 59)
                    .single()
                    .expect("invalid datetime"),
                notification_date: chrono::Utc
                    .with_ymd_and_hms(edition_year, 1, 15, 12, 0, 0)
                    .single()
                    .expect("invalid datetime"),
                // Feb 28 rather than 29: this is a synthetic template date
                // (not a specific historical deadline), and `edition_year`
                // varies at runtime, so it must stay valid on non-leap years.
                camera_ready_deadline: chrono::Utc
                    .with_ymd_and_hms(edition_year, 2, 28, 23, 59, 59)
                    .single()
                    .expect("invalid datetime"),
                conference_start: chrono::Utc
                    .with_ymd_and_hms(edition_year, 5, 7, 9, 0, 0)
                    .single()
                    .expect("invalid datetime"),
                conference_end: chrono::Utc
                    .with_ymd_and_hms(edition_year, 5, 11, 18, 0, 0)
                    .single()
                    .expect("invalid datetime"),
            },
            requirements: SubmissionRequirements {
                page_limit: 9,
                word_limit: None,
                format: FormatRequirements {
                    template: format!("ICLR {edition_year} LaTeX template"),
                    font_size: 10,
                    line_spacing: 1.0,
                    margins: "1 inch".to_string(),
                    citation_style: "ICLR".to_string(),
                    file_format: vec!["PDF".to_string()],
                },
                required_sections: vec![
                    "Abstract".to_string(),
                    "Introduction".to_string(),
                    "Related Work".to_string(),
                    "Method".to_string(),
                    "Experiments".to_string(),
                    "Conclusion".to_string(),
                ],
                supplementary_allowed: true,
                anonymous_submission: true,
                double_blind: true,
            },
            review_process: ReviewProcess {
                reviewers_per_paper: 3,
                review_criteria: vec![
                    "Technical Quality".to_string(),
                    "Clarity".to_string(),
                    "Originality".to_string(),
                    "Significance".to_string(),
                ],
                rebuttal_allowed: true,
                acceptance_rate: Some(0.31), // Approximately 31%
                review_format: ReviewFormat::Mixed,
            },
        }
    }

    fn create_aaai_conference(edition_year: i32) -> Conference {
        let prior_year = edition_year - 1;
        Conference {
            id: format!("aaai{edition_year}"),
            name: "AAAI Conference on Artificial Intelligence".to_string(),
            abbreviation: "AAAI".to_string(),
            description: "Conference on artificial intelligence".to_string(),
            url: "https://aaai.org/".to_string(),
            ranking: ConferenceRanking::TopTier,
            research_areas: vec![
                "Artificial Intelligence".to_string(),
                "Machine Learning".to_string(),
                "Knowledge Representation".to_string(),
                "Planning".to_string(),
            ],
            annual: true,
            series_info: SeriesInfo {
                series_number: 38,
                year: edition_year as u32,
                location: "Vancouver".to_string(),
                country: "Canada".to_string(),
                format: ConferenceFormat::Hybrid,
            },
            dates: ConferenceDates {
                abstract_deadline: Some(
                    chrono::Utc
                        .with_ymd_and_hms(prior_year, 8, 15, 23, 59, 59)
                        .single()
                        .expect("invalid datetime"),
                ),
                paper_deadline: chrono::Utc
                    .with_ymd_and_hms(prior_year, 8, 19, 23, 59, 59)
                    .single()
                    .expect("invalid datetime"),
                notification_date: chrono::Utc
                    .with_ymd_and_hms(prior_year, 12, 9, 12, 0, 0)
                    .single()
                    .expect("invalid datetime"),
                camera_ready_deadline: chrono::Utc
                    .with_ymd_and_hms(edition_year, 1, 15, 23, 59, 59)
                    .single()
                    .expect("invalid datetime"),
                conference_start: chrono::Utc
                    .with_ymd_and_hms(edition_year, 2, 20, 9, 0, 0)
                    .single()
                    .expect("invalid datetime"),
                conference_end: chrono::Utc
                    .with_ymd_and_hms(edition_year, 2, 27, 18, 0, 0)
                    .single()
                    .expect("invalid datetime"),
            },
            requirements: SubmissionRequirements {
                page_limit: 7,
                word_limit: None,
                format: FormatRequirements {
                    template: format!("AAAI {edition_year} LaTeX template"),
                    font_size: 10,
                    line_spacing: 1.0,
                    margins: "0.75 inch".to_string(),
                    citation_style: "AAAI".to_string(),
                    file_format: vec!["PDF".to_string()],
                },
                required_sections: vec![
                    "Abstract".to_string(),
                    "Introduction".to_string(),
                    "Related Work".to_string(),
                    "Approach".to_string(),
                    "Experiments".to_string(),
                    "Conclusion".to_string(),
                ],
                supplementary_allowed: false,
                anonymous_submission: true,
                double_blind: true,
            },
            review_process: ReviewProcess {
                reviewers_per_paper: 3,
                review_criteria: vec![
                    "Technical Quality".to_string(),
                    "Novelty".to_string(),
                    "Significance".to_string(),
                    "Clarity".to_string(),
                ],
                rebuttal_allowed: false,
                acceptance_rate: Some(0.23), // Approximately 23%
                review_format: ReviewFormat::NumericalScores,
            },
        }
    }

    fn create_ijcai_conference(edition_year: i32) -> Conference {
        Conference {
            id: format!("ijcai{edition_year}"),
            name: "International Joint Conference on Artificial Intelligence".to_string(),
            abbreviation: "IJCAI".to_string(),
            description: "International conference on artificial intelligence".to_string(),
            url: "https://ijcai.org/".to_string(),
            ranking: ConferenceRanking::TopTier,
            research_areas: vec![
                "Artificial Intelligence".to_string(),
                "Machine Learning".to_string(),
                "Automated Reasoning".to_string(),
                "Multi-agent Systems".to_string(),
            ],
            annual: true,
            series_info: SeriesInfo {
                series_number: 33,
                year: edition_year as u32,
                location: "Jeju".to_string(),
                country: "South Korea".to_string(),
                format: ConferenceFormat::Hybrid,
            },
            dates: ConferenceDates {
                abstract_deadline: Some(
                    chrono::Utc
                        .with_ymd_and_hms(edition_year, 1, 17, 23, 59, 59)
                        .single()
                        .expect("invalid datetime"),
                ),
                paper_deadline: chrono::Utc
                    .with_ymd_and_hms(edition_year, 1, 24, 23, 59, 59)
                    .single()
                    .expect("invalid datetime"),
                notification_date: chrono::Utc
                    .with_ymd_and_hms(edition_year, 4, 16, 12, 0, 0)
                    .single()
                    .expect("invalid datetime"),
                camera_ready_deadline: chrono::Utc
                    .with_ymd_and_hms(edition_year, 5, 15, 23, 59, 59)
                    .single()
                    .expect("invalid datetime"),
                conference_start: chrono::Utc
                    .with_ymd_and_hms(edition_year, 8, 3, 9, 0, 0)
                    .single()
                    .expect("invalid datetime"),
                conference_end: chrono::Utc
                    .with_ymd_and_hms(edition_year, 8, 9, 18, 0, 0)
                    .single()
                    .expect("invalid datetime"),
            },
            requirements: SubmissionRequirements {
                page_limit: 7,
                word_limit: None,
                format: FormatRequirements {
                    template: format!("IJCAI {edition_year} LaTeX template"),
                    font_size: 10,
                    line_spacing: 1.0,
                    margins: "0.75 inch".to_string(),
                    citation_style: "IJCAI".to_string(),
                    file_format: vec!["PDF".to_string()],
                },
                required_sections: vec![
                    "Abstract".to_string(),
                    "Introduction".to_string(),
                    "Background".to_string(),
                    "Approach".to_string(),
                    "Experiments".to_string(),
                    "Conclusion".to_string(),
                ],
                supplementary_allowed: false,
                anonymous_submission: true,
                double_blind: true,
            },
            review_process: ReviewProcess {
                reviewers_per_paper: 3,
                review_criteria: vec![
                    "Technical Quality".to_string(),
                    "Novelty".to_string(),
                    "Significance".to_string(),
                    "Clarity".to_string(),
                ],
                rebuttal_allowed: true,
                acceptance_rate: Some(0.15), // Approximately 15%
                review_format: ReviewFormat::Mixed,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conference_manager_creation() {
        let manager = ConferenceManager::new();
        assert!(manager.conferences.is_empty());
        assert!(manager.submissions.is_empty());
    }

    #[test]
    fn test_load_standard_conferences() {
        let mut manager = ConferenceManager::new();
        manager.load_standard_conferences();

        assert_eq!(manager.conferences.len(), 5);
        let abbreviations: Vec<&str> = manager
            .conferences
            .values()
            .map(|c| c.abbreviation.as_str())
            .collect();
        for expected in ["NeurIPS", "ICML", "ICLR", "AAAI", "IJCAI"] {
            assert!(
                abbreviations.contains(&expected),
                "missing {expected} in {abbreviations:?}"
            );
        }
    }

    // Regression test for F78: the standard conference templates had every
    // deadline hardcoded to a fixed calendar year (2023/2024). Once that
    // year passed, `get_upcoming_deadlines` could never return anything for
    // them again. `load_standard_conferences` must now always select an
    // edition whose deadlines lie in the future, however far "now" is from
    // when this code was written.
    #[test]
    fn test_load_standard_conferences_deadlines_are_in_the_future() {
        let mut manager = ConferenceManager::new();
        manager.load_standard_conferences();
        let now = Utc::now();

        for conference in manager.conferences.values() {
            assert!(
                conference.dates.paper_deadline > now,
                "{}'s paper deadline {} is not in the future (now = {now})",
                conference.abbreviation,
                conference.dates.paper_deadline
            );
        }

        // With a wide enough horizon, every loaded conference must surface
        // at least one upcoming deadline -- this was unconditionally empty
        // before the fix.
        let upcoming = manager.get_upcoming_deadlines(400);
        assert!(
            !upcoming.is_empty(),
            "expected at least one upcoming deadline within 400 days"
        );
    }

    // Regression test for F78: `alerts` was declared on `ConferenceManager`
    // but no code path ever wrote to it.
    #[test]
    fn test_generate_deadline_alerts_populates_alerts() {
        let mut manager = ConferenceManager::new();
        manager.load_standard_conferences();
        assert!(manager.alerts.is_empty());

        // A very wide threshold guarantees at least the paper deadlines
        // (already asserted to be in the future) fall inside the window.
        let created = manager.generate_deadline_alerts(&[400]);
        assert!(!created.is_empty());
        assert_eq!(manager.alerts.len(), created.len());
        assert!(manager.alerts.iter().all(|a| !a.sent));
        assert_eq!(manager.pending_alerts().len(), manager.alerts.len());

        // Idempotent: calling again with the same thresholds must not
        // duplicate alerts for the same (conference, type, threshold).
        let created_again = manager.generate_deadline_alerts(&[400]);
        assert!(created_again.is_empty());
        assert_eq!(manager.alerts.len(), created.len());

        let alert_id = manager.alerts[0].id.clone();
        manager
            .mark_alert_sent(&alert_id)
            .expect("mark should succeed");
        assert!(
            manager
                .alerts
                .iter()
                .find(|a| a.id == alert_id)
                .unwrap()
                .sent
        );
        assert_eq!(manager.pending_alerts().len(), manager.alerts.len() - 1);
    }

    #[test]
    fn test_search_conferences() {
        let mut manager = ConferenceManager::new();
        manager.load_standard_conferences();

        let ml_conferences = manager.search_conferences("Machine Learning");
        assert!(!ml_conferences.is_empty());

        let top_tier = manager.get_conferences_by_ranking(ConferenceRanking::TopTier);
        assert!(!top_tier.is_empty());
    }
}
