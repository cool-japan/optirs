// Peer review tools and anonymous review systems
//
// This module provides tools for managing peer review processes,
// including anonymous reviews, reviewer assignment, and review quality assessment.

use crate::error::{OptimError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Peer review system manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerReviewSystem {
    /// Review sessions
    pub sessions: HashMap<String, ReviewSession>,
    /// Reviewer pool
    pub reviewers: HashMap<String, Reviewer>,
    /// Review assignments
    pub assignments: Vec<ReviewAssignment>,
    /// Review quality metrics
    pub quality_metrics: Vec<ReviewQualityMetric>,
}

/// Review session for a paper or project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewSession {
    /// Session ID
    pub id: String,
    /// Paper/project ID
    pub submission_id: String,
    /// Review type
    pub review_type: ReviewType,
    /// Session status
    pub status: ReviewSessionStatus,
    /// Review criteria
    pub criteria: Vec<ReviewCriterion>,
    /// Deadline
    pub deadline: DateTime<Utc>,
    /// Reviews collected
    pub reviews: Vec<PeerReview>,
    /// Meta-review
    pub meta_review: Option<MetaReview>,
    /// Discussion thread
    pub discussion: Vec<ReviewDiscussion>,
}

/// Types of peer review
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReviewType {
    /// Single-blind review
    SingleBlind,
    /// Double-blind review
    DoubleBlind,
    /// Open review
    Open,
    /// Post-publication review
    PostPublication,
    /// Internal review
    Internal,
}

/// Review session status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReviewSessionStatus {
    /// Waiting for reviewers
    WaitingForReviewers,
    /// Reviews in progress
    InProgress,
    /// Reviews complete
    ReviewsComplete,
    /// Meta-review in progress
    MetaReviewInProgress,
    /// Session complete
    Complete,
    /// Session cancelled
    Cancelled,
}

/// Review criterion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewCriterion {
    /// Criterion name
    pub name: String,
    /// Description
    pub description: String,
    /// Score range
    pub score_range: (f64, f64),
    /// Weight in overall score
    pub weight: f64,
    /// Required for review
    pub required: bool,
}

/// Individual peer review
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerReview {
    /// Review ID
    pub id: String,
    /// Anonymous reviewer ID
    pub reviewer_id: String,
    /// Overall recommendation
    pub recommendation: ReviewRecommendation,
    /// Scores per criterion
    pub criterion_scores: HashMap<String, f64>,
    /// Overall score
    pub overall_score: f64,
    /// Confidence level
    pub confidence: f64,
    /// Written review
    pub written_review: WrittenReview,
    /// Review status
    pub status: ReviewStatus,
    /// Submission timestamp
    pub submitted_at: Option<DateTime<Utc>>,
    /// Time spent on review (minutes)
    pub time_spent_minutes: Option<u32>,
}

/// Review recommendations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ReviewRecommendation {
    /// Strong accept
    StrongAccept,
    /// Accept
    Accept,
    /// Weak accept
    WeakAccept,
    /// Borderline accept
    BorderlineAccept,
    /// Borderline reject
    BorderlineReject,
    /// Weak reject
    WeakReject,
    /// Reject
    Reject,
    /// Strong reject
    StrongReject,
}

/// Written review components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WrittenReview {
    /// Summary
    pub summary: String,
    /// Strengths
    pub strengths: Vec<String>,
    /// Weaknesses
    pub weaknesses: Vec<String>,
    /// Detailed comments
    pub detailed_comments: String,
    /// Questions for authors
    pub questions: Vec<String>,
    /// Minor issues
    pub minor_issues: Vec<String>,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
    /// Comments for committee only
    pub committee_comments: Option<String>,
}

/// Review status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReviewStatus {
    /// Assigned but not started
    Assigned,
    /// In progress
    InProgress,
    /// Draft completed
    Draft,
    /// Submitted
    Submitted,
    /// Revision requested
    RevisionRequested,
    /// Declined
    Declined,
    /// Overdue
    Overdue,
}

/// Meta-review (review of reviews)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaReview {
    /// Meta-reviewer ID
    pub meta_reviewer_id: String,
    /// Summary of individual reviews
    pub review_summary: String,
    /// Final recommendation
    pub final_recommendation: ReviewRecommendation,
    /// Justification
    pub justification: String,
    /// Review quality assessment
    pub review_quality: Vec<ReviewQualityAssessment>,
    /// Areas of agreement
    pub areas_of_agreement: Vec<String>,
    /// Areas of disagreement
    pub areas_of_disagreement: Vec<String>,
    /// Decision rationale
    pub decision_rationale: String,
}

/// Assessment of individual review quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewQualityAssessment {
    /// Review ID
    pub review_id: String,
    /// Quality dimensions
    pub quality_scores: HashMap<String, f64>,
    /// Overall quality score
    pub overall_quality: f64,
    /// Helpfulness to authors
    pub helpfulness: f64,
    /// Comments on review quality
    pub comments: String,
}

/// Review discussion thread
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewDiscussion {
    /// Post ID
    pub id: String,
    /// Author (anonymous)
    pub author: String,
    /// Post content
    pub content: String,
    /// Reply to post ID
    pub reply_to: Option<String>,
    /// Post timestamp
    pub posted_at: DateTime<Utc>,
    /// Post type
    pub post_type: DiscussionPostType,
}

/// Types of discussion posts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DiscussionPostType {
    /// Question
    Question,
    /// Answer
    Answer,
    /// Clarification
    Clarification,
    /// Disagreement
    Disagreement,
    /// Consensus
    Consensus,
    /// Moderator message
    ModeratorMessage,
}

/// Reviewer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reviewer {
    /// Anonymous reviewer ID
    pub id: String,
    /// Expertise areas
    pub expertise_areas: Vec<String>,
    /// Experience level
    pub experience_level: ExperienceLevel,
    /// Review history
    pub review_history: ReviewerHistory,
    /// Availability
    pub availability: ReviewerAvailability,
    /// Quality metrics
    pub quality_metrics: ReviewerQualityMetrics,
    /// Preferences
    pub preferences: ReviewerPreferences,
}

/// Reviewer experience levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExperienceLevel {
    /// Expert reviewer
    Expert,
    /// Senior reviewer
    Senior,
    /// Experienced reviewer
    Experienced,
    /// Junior reviewer
    Junior,
    /// Novice reviewer
    Novice,
}

/// Reviewer history and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewerHistory {
    /// Total reviews completed
    pub total_reviews: usize,
    /// Reviews in last 12 months
    pub reviews_last_year: usize,
    /// Average review time (days)
    pub avg_review_time_days: f64,
    /// On-time submission rate
    pub on_time_rate: f64,
    /// Average review quality score
    pub avg_quality_score: f64,
    /// Review acceptance rate
    pub review_acceptance_rate: f64,
}

/// Reviewer availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewerAvailability {
    /// Currently available
    pub available: bool,
    /// Maximum reviews per month
    pub max_reviews_per_month: u32,
    /// Current review load
    pub current_load: u32,
    /// Unavailable periods
    pub unavailable_periods: Vec<(DateTime<Utc>, DateTime<Utc>)>,
    /// Preferred review types
    pub preferred_types: Vec<ReviewType>,
}

/// Reviewer quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewerQualityMetrics {
    /// Thoroughness score
    pub thoroughness: f64,
    /// Constructiveness score
    pub constructiveness: f64,
    /// Timeliness score
    pub timeliness: f64,
    /// Expertise match score
    pub expertise_match: f64,
    /// Overall reviewer score
    pub overall_score: f64,
}

/// Reviewer preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewerPreferences {
    /// Preferred paper types
    pub preferred_paper_types: Vec<String>,
    /// Avoid paper types
    pub avoid_paper_types: Vec<String>,
    /// Maximum review length preference
    pub max_review_length: Option<u32>,
    /// Anonymous review preference
    pub anonymous_preference: bool,
    /// Notification preferences
    pub notification_preferences: NotificationPreferences,
}

/// Notification preferences for reviewers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPreferences {
    /// Email notifications
    pub email: bool,
    /// Reminder frequency (days)
    pub reminder_frequency: u32,
    /// Deadline notifications
    pub deadline_notifications: bool,
    /// Discussion notifications
    pub discussion_notifications: bool,
}

/// Review assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewAssignment {
    /// Assignment ID
    pub id: String,
    /// Session ID
    pub session_id: String,
    /// Reviewer ID
    pub reviewer_id: String,
    /// Assignment date
    pub assigned_at: DateTime<Utc>,
    /// Due date
    pub due_date: DateTime<Utc>,
    /// Assignment status
    pub status: AssignmentStatus,
    /// Assignment method
    pub assignment_method: AssignmentMethod,
    /// Expertise match score
    pub expertise_match: f64,
}

/// Assignment status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AssignmentStatus {
    /// Pending acceptance
    Pending,
    /// Accepted
    Accepted,
    /// Declined
    Declined,
    /// Completed
    Completed,
    /// Overdue
    Overdue,
    /// Cancelled
    Cancelled,
}

/// Assignment methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AssignmentMethod {
    /// Manual assignment
    Manual,
    /// Automatic based on expertise
    AutomaticExpertise,
    /// Automatic load balancing
    AutomaticLoadBalancing,
    /// Hybrid assignment
    Hybrid,
    /// Self-assignment
    SelfAssignment,
}

/// Review quality metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewQualityMetric {
    /// Metric name
    pub name: String,
    /// Description
    pub description: String,
    /// Value range
    pub value_range: (f64, f64),
    /// Higher is better
    pub higher_is_better: bool,
    /// Calculation method
    pub calculation_method: String,
}

impl PeerReviewSystem {
    /// Create a new peer review system
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            reviewers: HashMap::new(),
            assignments: Vec::new(),
            quality_metrics: Self::create_default_quality_metrics(),
        }
    }

    /// Create a new review session
    pub fn create_review_session(
        &mut self,
        submission_id: &str,
        review_type: ReviewType,
        criteria: Vec<ReviewCriterion>,
        deadline: DateTime<Utc>,
    ) -> String {
        let session_id = uuid::Uuid::new_v4().to_string();
        let session = ReviewSession {
            id: session_id.clone(),
            submission_id: submission_id.to_string(),
            review_type,
            status: ReviewSessionStatus::WaitingForReviewers,
            criteria,
            deadline,
            reviews: Vec::new(),
            meta_review: None,
            discussion: Vec::new(),
        };

        self.sessions.insert(session_id.clone(), session);
        session_id
    }

    /// Assign reviewers to a session
    pub fn assign_reviewers(
        &mut self,
        session_id: &str,
        reviewer_ids: &[String],
        assignment_method: AssignmentMethod,
    ) -> Result<Vec<String>> {
        if !self.sessions.contains_key(session_id) {
            return Err(OptimError::InvalidConfig("Session not found".to_string()));
        }

        let mut assignment_ids = Vec::new();
        let now = Utc::now();
        let session = self.sessions.get(session_id).unwrap();

        for reviewer_id in reviewer_ids {
            if !self.reviewers.contains_key(reviewer_id) {
                continue; // Skip unknown reviewers
            }

            let assignment_id = uuid::Uuid::new_v4().to_string();
            let assignment = ReviewAssignment {
                id: assignment_id.clone(),
                session_id: session_id.to_string(),
                reviewer_id: reviewer_id.clone(),
                assigned_at: now,
                due_date: session.deadline,
                status: AssignmentStatus::Pending,
                assignment_method: assignment_method.clone(),
                expertise_match: self.calculate_expertise_match(reviewer_id, session_id),
            };

            self.assignments.push(assignment);
            assignment_ids.push(assignment_id);
        }

        // Update session status
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.status = ReviewSessionStatus::InProgress;
        }

        Ok(assignment_ids)
    }

    /// Submit a peer review
    pub fn submit_review(
        &mut self,
        session_id: &str,
        reviewer_id: &str,
        review: PeerReview,
    ) -> Result<()> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| OptimError::InvalidConfig("Session not found".to_string()))?;

        // Verify reviewer is assigned
        let assignment = self
            .assignments
            .iter_mut()
            .find(|a| a.session_id == session_id && a.reviewer_id == reviewer_id)
            .ok_or_else(|| {
                OptimError::InvalidConfig("Reviewer not assigned to this session".to_string())
            })?;

        // Update assignment status
        assignment.status = AssignmentStatus::Completed;

        // Add review to session
        session.reviews.push(review);

        // Check if all reviews are complete
        let total_assignments = self
            .assignments
            .iter()
            .filter(|a| a.session_id == session_id)
            .count();

        if session.reviews.len() == total_assignments {
            session.status = ReviewSessionStatus::ReviewsComplete;
        }

        Ok(())
    }

    /// Generate meta-review
    pub fn generate_meta_review(&mut self, session_id: &str, meta_reviewer_id: &str) -> Result<()> {
        // First, check session status and create meta review with immutable access
        let meta_review = {
            let session = self
                .sessions
                .get(session_id)
                .ok_or_else(|| OptimError::InvalidConfig("Session not found".to_string()))?;

            if session.status != ReviewSessionStatus::ReviewsComplete {
                return Err(OptimError::InvalidConfig(
                    "Not all reviews are complete".to_string(),
                ));
            }

            self.create_meta_review(session, meta_reviewer_id)
        };

        // Now update the session with mutable access
        let session = self.sessions.get_mut(session_id).unwrap(); // Safe because we just checked it exists
        session.meta_review = Some(meta_review);
        session.status = ReviewSessionStatus::Complete;

        Ok(())
    }

    /// Calculate reviewer workload
    pub fn calculate_reviewer_workload(&self, reviewer_id: &str) -> u32 {
        self.assignments
            .iter()
            .filter(|a| {
                a.reviewer_id == reviewer_id
                    && matches!(
                        a.status,
                        AssignmentStatus::Pending | AssignmentStatus::Accepted
                    )
            })
            .count() as u32
    }

    /// Get available reviewers for expertise area
    pub fn get_available_reviewers(&self, expertise_area: &str) -> Vec<&Reviewer> {
        self.reviewers
            .values()
            .filter(|r| {
                r.availability.available
                    && r.expertise_areas
                        .iter()
                        .any(|area| area.to_lowercase().contains(&expertise_area.to_lowercase()))
                    && r.availability.current_load < r.availability.max_reviews_per_month
            })
            .collect()
    }

    /// Calculate review quality score
    pub fn calculate_review_quality(&self, review: &PeerReview) -> f64 {
        let mut quality_score = 0.0;
        let mut total_weight = 0.0;

        // Length and detail assessment
        let review_length = review.written_review.detailed_comments.len()
            + review
                .written_review
                .strengths
                .iter()
                .map(|s| s.len())
                .sum::<usize>()
            + review
                .written_review
                .weaknesses
                .iter()
                .map(|s| s.len())
                .sum::<usize>();

        let length_score = ((review_length as f64).ln() / 10.0).min(1.0);
        quality_score += length_score * 0.3;
        total_weight += 0.3;

        // Number of specific points
        let specific_points = review.written_review.strengths.len()
            + review.written_review.weaknesses.len()
            + review.written_review.suggestions.len();

        let specificity_score = (specific_points as f64 / 10.0).min(1.0);
        quality_score += specificity_score * 0.4;
        total_weight += 0.4;

        // Confidence level
        quality_score += review.confidence * 0.3;
        total_weight += 0.3;

        quality_score / total_weight
    }

    fn calculate_expertise_match(&self, reviewer_id: &str, sessionid: &str) -> f64 {
        // Simplified expertise matching
        // In practice, you'd use more sophisticated matching algorithms
        if let Some(reviewer) = self.reviewers.get(reviewer_id) {
            if reviewer.expertise_areas.is_empty() {
                0.5 // Default moderate match
            } else {
                0.8 // Good match if has expertise areas
            }
        } else {
            0.0
        }
    }

    fn create_meta_review(&self, session: &ReviewSession, meta_reviewer_id: &str) -> MetaReview {
        let review_summary = format!("Meta-review of {} reviews", session.reviews.len());

        // Calculate consensus
        let recommendations: Vec<_> = session.reviews.iter().map(|r| &r.recommendation).collect();

        let final_recommendation = self.determine_consensus_recommendation(&recommendations);

        // Assess review quality
        let review_quality: Vec<_> = session
            .reviews
            .iter()
            .map(|review| {
                let quality_score = self.calculate_review_quality(review);
                ReviewQualityAssessment {
                    review_id: review.id.clone(),
                    quality_scores: HashMap::new(),
                    overall_quality: quality_score,
                    helpfulness: quality_score * 0.9, // Simplified
                    comments: if quality_score > 0.7 {
                        "High quality review".to_string()
                    } else {
                        "Review could be more detailed".to_string()
                    },
                }
            })
            .collect();

        MetaReview {
            meta_reviewer_id: meta_reviewer_id.to_string(),
            review_summary,
            final_recommendation,
            justification: "Based on consensus of reviewer recommendations".to_string(),
            review_quality,
            areas_of_agreement: vec!["Technical quality assessment".to_string()],
            areas_of_disagreement: vec!["Significance of contribution".to_string()],
            decision_rationale: "Decision based on majority reviewer consensus".to_string(),
        }
    }

    fn determine_consensus_recommendation(
        &self,
        recommendations: &[&ReviewRecommendation],
    ) -> ReviewRecommendation {
        // Simplified consensus algorithm - use majority vote
        let mut counts = HashMap::new();
        for rec in recommendations {
            *counts.entry(rec).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(rec, _)| rec.clone())
            .unwrap_or(&ReviewRecommendation::BorderlineReject)
            .clone()
    }

    fn create_default_quality_metrics() -> Vec<ReviewQualityMetric> {
        vec![
            ReviewQualityMetric {
                name: "Thoroughness".to_string(),
                description: "How comprehensive and detailed the review is".to_string(),
                value_range: (0.0, 1.0),
                higher_is_better: true,
                calculation_method: "Based on review length and number of specific points"
                    .to_string(),
            },
            ReviewQualityMetric {
                name: "Constructiveness".to_string(),
                description: "How helpful the review is for improving the work".to_string(),
                value_range: (0.0, 1.0),
                higher_is_better: true,
                calculation_method: "Based on number of suggestions and actionable feedback"
                    .to_string(),
            },
            ReviewQualityMetric {
                name: "Timeliness".to_string(),
                description: "How promptly the review was submitted".to_string(),
                value_range: (0.0, 1.0),
                higher_is_better: true,
                calculation_method: "Based on submission time relative to deadline".to_string(),
            },
        ]
    }
}

impl Default for ReviewerHistory {
    fn default() -> Self {
        Self {
            total_reviews: 0,
            reviews_last_year: 0,
            avg_review_time_days: 14.0,
            on_time_rate: 1.0,
            avg_quality_score: 0.7,
            review_acceptance_rate: 0.9,
        }
    }
}

impl Default for ReviewerAvailability {
    fn default() -> Self {
        Self {
            available: true,
            max_reviews_per_month: 5,
            current_load: 0,
            unavailable_periods: Vec::new(),
            preferred_types: vec![ReviewType::DoubleBlind],
        }
    }
}

impl Default for ReviewerQualityMetrics {
    fn default() -> Self {
        Self {
            thoroughness: 0.7,
            constructiveness: 0.7,
            timeliness: 0.8,
            expertise_match: 0.7,
            overall_score: 0.7,
        }
    }
}

impl Default for NotificationPreferences {
    fn default() -> Self {
        Self {
            email: true,
            reminder_frequency: 7,
            deadline_notifications: true,
            discussion_notifications: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_review_system_creation() {
        let system = PeerReviewSystem::new();
        assert!(system.sessions.is_empty());
        assert!(system.reviewers.is_empty());
        assert!(!system.quality_metrics.is_empty());
    }

    #[test]
    fn test_create_review_session() {
        let mut system = PeerReviewSystem::new();

        let criteria = vec![ReviewCriterion {
            name: "Technical Quality".to_string(),
            description: "Assessment of technical merit".to_string(),
            score_range: (1.0, 5.0),
            weight: 0.4,
            required: true,
        }];

        let deadline = Utc::now() + chrono::Duration::days(14);
        let session_id =
            system.create_review_session("paper123", ReviewType::DoubleBlind, criteria, deadline);

        assert!(system.sessions.contains_key(&session_id));
        let session = &system.sessions[&session_id];
        assert_eq!(session.submission_id, "paper123");
        assert_eq!(session.review_type, ReviewType::DoubleBlind);
    }

    #[test]
    fn test_reviewer_workload_calculation() {
        let mut system = PeerReviewSystem::new();

        // Add some assignments
        system.assignments.push(ReviewAssignment {
            id: "assign1".to_string(),
            session_id: "session1".to_string(),
            reviewer_id: "reviewer1".to_string(),
            assigned_at: Utc::now(),
            due_date: Utc::now() + chrono::Duration::days(14),
            status: AssignmentStatus::Pending,
            assignment_method: AssignmentMethod::Manual,
            expertise_match: 0.8,
        });

        let workload = system.calculate_reviewer_workload("reviewer1");
        assert_eq!(workload, 1);

        let workload = system.calculate_reviewer_workload("reviewer2");
        assert_eq!(workload, 0);
    }
}
