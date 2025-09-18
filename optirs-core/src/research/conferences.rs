// Academic conference integration and submission tools
//
// This module provides tools for managing conference submissions,
// tracking deadlines, and preparing submission materials.

use crate::error::{OptimError, Result};
use chrono::{DateTime, TimeZone, Utc};
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
        deadlines.sort_by(|a, b| a.2.cmp(&b.2));
        deadlines
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

    /// Create standard ML/AI conferences
    pub fn load_standard_conferences(&mut self) {
        // Add some well-known conferences
        self.add_conference(Self::create_neurips_conference());
        self.add_conference(Self::create_icml_conference());
        self.add_conference(Self::create_iclr_conference());
        self.add_conference(Self::create_aaai_conference());
        self.add_conference(Self::create_ijcai_conference());
    }

    fn create_neurips_conference() -> Conference {
        Conference {
            id: "neurips2024".to_string(),
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
                year: 2024,
                location: "Vancouver".to_string(),
                country: "Canada".to_string(),
                format: ConferenceFormat::Hybrid,
            },
            dates: ConferenceDates {
                abstract_deadline: Some(
                    chrono::Utc
                        .with_ymd_and_hms(2024, 5, 15, 23, 59, 59)
                        .unwrap(),
                ),
                paper_deadline: chrono::Utc
                    .with_ymd_and_hms(2024, 5, 22, 23, 59, 59)
                    .unwrap(),
                notification_date: chrono::Utc.with_ymd_and_hms(2024, 9, 25, 12, 0, 0).unwrap(),
                camera_ready_deadline: chrono::Utc
                    .with_ymd_and_hms(2024, 10, 30, 23, 59, 59)
                    .unwrap(),
                conference_start: chrono::Utc.with_ymd_and_hms(2024, 12, 10, 9, 0, 0).unwrap(),
                conference_end: chrono::Utc
                    .with_ymd_and_hms(2024, 12, 16, 18, 0, 0)
                    .unwrap(),
            },
            requirements: SubmissionRequirements {
                page_limit: 9,
                word_limit: None,
                format: FormatRequirements {
                    template: "NeurIPS 2024 LaTeX template".to_string(),
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

    fn create_icml_conference() -> Conference {
        Conference {
            id: "icml2024".to_string(),
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
                year: 2024,
                location: "Vienna".to_string(),
                country: "Austria".to_string(),
                format: ConferenceFormat::Hybrid,
            },
            dates: ConferenceDates {
                abstract_deadline: None,
                paper_deadline: chrono::Utc
                    .with_ymd_and_hms(2024, 2, 1, 23, 59, 59)
                    .unwrap(),
                notification_date: chrono::Utc.with_ymd_and_hms(2024, 5, 1, 12, 0, 0).unwrap(),
                camera_ready_deadline: chrono::Utc
                    .with_ymd_and_hms(2024, 6, 1, 23, 59, 59)
                    .unwrap(),
                conference_start: chrono::Utc.with_ymd_and_hms(2024, 7, 21, 9, 0, 0).unwrap(),
                conference_end: chrono::Utc.with_ymd_and_hms(2024, 7, 27, 18, 0, 0).unwrap(),
            },
            requirements: SubmissionRequirements {
                page_limit: 8,
                word_limit: None,
                format: FormatRequirements {
                    template: "ICML 2024 LaTeX template".to_string(),
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

    fn create_iclr_conference() -> Conference {
        Conference {
            id: "iclr2024".to_string(),
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
                year: 2024,
                location: "Vienna".to_string(),
                country: "Austria".to_string(),
                format: ConferenceFormat::Hybrid,
            },
            dates: ConferenceDates {
                abstract_deadline: Some(
                    chrono::Utc
                        .with_ymd_and_hms(2023, 9, 28, 23, 59, 59)
                        .unwrap(),
                ),
                paper_deadline: chrono::Utc
                    .with_ymd_and_hms(2023, 10, 2, 23, 59, 59)
                    .unwrap(),
                notification_date: chrono::Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap(),
                camera_ready_deadline: chrono::Utc
                    .with_ymd_and_hms(2024, 2, 29, 23, 59, 59)
                    .unwrap(),
                conference_start: chrono::Utc.with_ymd_and_hms(2024, 5, 7, 9, 0, 0).unwrap(),
                conference_end: chrono::Utc.with_ymd_and_hms(2024, 5, 11, 18, 0, 0).unwrap(),
            },
            requirements: SubmissionRequirements {
                page_limit: 9,
                word_limit: None,
                format: FormatRequirements {
                    template: "ICLR 2024 LaTeX template".to_string(),
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

    fn create_aaai_conference() -> Conference {
        Conference {
            id: "aaai2024".to_string(),
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
                year: 2024,
                location: "Vancouver".to_string(),
                country: "Canada".to_string(),
                format: ConferenceFormat::Hybrid,
            },
            dates: ConferenceDates {
                abstract_deadline: Some(
                    chrono::Utc
                        .with_ymd_and_hms(2023, 8, 15, 23, 59, 59)
                        .unwrap(),
                ),
                paper_deadline: chrono::Utc
                    .with_ymd_and_hms(2023, 8, 19, 23, 59, 59)
                    .unwrap(),
                notification_date: chrono::Utc.with_ymd_and_hms(2023, 12, 9, 12, 0, 0).unwrap(),
                camera_ready_deadline: chrono::Utc
                    .with_ymd_and_hms(2024, 1, 15, 23, 59, 59)
                    .unwrap(),
                conference_start: chrono::Utc.with_ymd_and_hms(2024, 2, 20, 9, 0, 0).unwrap(),
                conference_end: chrono::Utc.with_ymd_and_hms(2024, 2, 27, 18, 0, 0).unwrap(),
            },
            requirements: SubmissionRequirements {
                page_limit: 7,
                word_limit: None,
                format: FormatRequirements {
                    template: "AAAI 2024 LaTeX template".to_string(),
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

    fn create_ijcai_conference() -> Conference {
        Conference {
            id: "ijcai2024".to_string(),
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
                year: 2024,
                location: "Jeju".to_string(),
                country: "South Korea".to_string(),
                format: ConferenceFormat::Hybrid,
            },
            dates: ConferenceDates {
                abstract_deadline: Some(
                    chrono::Utc
                        .with_ymd_and_hms(2024, 1, 17, 23, 59, 59)
                        .unwrap(),
                ),
                paper_deadline: chrono::Utc
                    .with_ymd_and_hms(2024, 1, 24, 23, 59, 59)
                    .unwrap(),
                notification_date: chrono::Utc.with_ymd_and_hms(2024, 4, 16, 12, 0, 0).unwrap(),
                camera_ready_deadline: chrono::Utc
                    .with_ymd_and_hms(2024, 5, 15, 23, 59, 59)
                    .unwrap(),
                conference_start: chrono::Utc.with_ymd_and_hms(2024, 8, 3, 9, 0, 0).unwrap(),
                conference_end: chrono::Utc.with_ymd_and_hms(2024, 8, 9, 18, 0, 0).unwrap(),
            },
            requirements: SubmissionRequirements {
                page_limit: 7,
                word_limit: None,
                format: FormatRequirements {
                    template: "IJCAI 2024 LaTeX template".to_string(),
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

        assert!(manager.conferences.contains_key("neurips2024"));
        assert!(manager.conferences.contains_key("icml2024"));
        assert!(manager.conferences.contains_key("iclr2024"));
        assert!(manager.conferences.contains_key("aaai2024"));
        assert!(manager.conferences.contains_key("ijcai2024"));
    }

    #[test]
    fn test_search_conferences() {
        let mut manager = ConferenceManager::new();
        manager.load_standard_conferences();

        let ml_conferences = manager.search_conferences("Machine Learning");
        assert!(ml_conferences.len() > 0);

        let top_tier = manager.get_conferences_by_ranking(ConferenceRanking::TopTier);
        assert!(top_tier.len() > 0);
    }
}
