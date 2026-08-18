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

impl Default for PeerReviewSystem {
    fn default() -> Self {
        Self::new()
    }
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

    /// Assign reviewers to a session.
    ///
    /// Skips any reviewer who is unknown or who already has an active
    /// (pending/accepted/completed) assignment for this session -- without
    /// this check, calling this twice with overlapping reviewer lists (or
    /// passing a duplicate ID) created two assignments for the same
    /// (session, reviewer) pair, which corrupted the reviews-complete count
    /// in [`Self::submit_review`] and double-counted reviewer workload.
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
        let deadline = self
            .sessions
            .get(session_id)
            .ok_or_else(|| {
                OptimError::InvalidState(format!("session '{session_id}' vanished during lookup"))
            })?
            .deadline;

        for reviewer_id in reviewer_ids {
            if !self.reviewers.contains_key(reviewer_id) {
                continue; // Skip unknown reviewers
            }

            let already_assigned = self
                .assignments
                .iter()
                .any(|a| a.session_id == session_id && a.reviewer_id == *reviewer_id);
            if already_assigned {
                continue;
            }

            let assignment_id = uuid::Uuid::new_v4().to_string();
            let assignment = ReviewAssignment {
                id: assignment_id.clone(),
                session_id: session_id.to_string(),
                reviewer_id: reviewer_id.clone(),
                assigned_at: now,
                due_date: deadline,
                status: AssignmentStatus::Pending,
                assignment_method: assignment_method.clone(),
                expertise_match: self.calculate_expertise_match(reviewer_id, session_id),
            };

            self.assignments.push(assignment);
            assignment_ids.push(assignment_id);

            // Track load on the reviewer record itself: this is what
            // `get_available_reviewers` consults to avoid overloading a
            // reviewer, so it must reflect assignments as they happen.
            if let Some(reviewer) = self.reviewers.get_mut(reviewer_id) {
                reviewer.availability.current_load += 1;
            }
        }

        // Update session status
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.status = ReviewSessionStatus::InProgress;
        }

        Ok(assignment_ids)
    }

    /// Submit a peer review.
    ///
    /// Rejects a second submission from the same reviewer for the same
    /// session: without this check, a duplicate call inflated
    /// `session.reviews` past the number of distinct assigned reviewers,
    /// which both skewed the meta-review consensus (the same opinion
    /// counted twice) and could prevent `reviews.len() == total_assignments`
    /// from ever matching (so the session would never reach
    /// `ReviewsComplete`).
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

        if assignment.status == AssignmentStatus::Completed {
            return Err(OptimError::InvalidConfig(format!(
                "reviewer '{reviewer_id}' already submitted a review for session '{session_id}'"
            )));
        }

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

        // A completed review is no longer part of the reviewer's *active*
        // load (mirrors `calculate_reviewer_workload`, which only counts
        // Pending/Accepted assignments), freeing capacity for new
        // assignments now that this one is done.
        if let Some(reviewer) = self.reviewers.get_mut(reviewer_id) {
            reviewer.availability.current_load =
                reviewer.availability.current_load.saturating_sub(1);
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

        // Now update the session with mutable access.
        let session = self.sessions.get_mut(session_id).ok_or_else(|| {
            OptimError::InvalidState(format!("session '{session_id}' vanished during lookup"))
        })?;
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

        // `ln_1p` (ln(1+x)) rather than `ln(x)`: a review with zero
        // measured length (empty comments/strengths/weaknesses) is a
        // realistic, valid input, and `ln(0) == -inf` would otherwise poison
        // the entire quality score to `-inf` (see regression test below).
        let length_score = ((review_length as f64).ln_1p() / 10.0).clamp(0.0, 1.0);
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

    /// How well a reviewer's declared expertise covers a session's criteria, in
    /// `[0, 1]`.
    ///
    /// The score is the fraction of the session's review criteria whose name or
    /// description mentions one of the reviewer's expertise areas
    /// (case-insensitive substring match). A reviewer with no declared expertise
    /// scores `0.0`, and a session with no criteria yields `0.5` -- there is
    /// nothing to match against, so neither a good nor a bad match can be
    /// claimed.
    ///
    /// Until 0.3.2 this ignored `sessionid` entirely and returned the constant
    /// `0.8` for any reviewer with a non-empty `expertise_areas` list and `0.5`
    /// otherwise, so assignment ranked a cryptographer and a numerical analyst
    /// identically on an optimization paper.
    fn calculate_expertise_match(&self, reviewer_id: &str, sessionid: &str) -> f64 {
        let Some(reviewer) = self.reviewers.get(reviewer_id) else {
            return 0.0;
        };
        if reviewer.expertise_areas.is_empty() {
            return 0.0;
        }
        let Some(session) = self.sessions.get(sessionid) else {
            return 0.5;
        };
        if session.criteria.is_empty() {
            return 0.5;
        }

        let areas: Vec<String> = reviewer
            .expertise_areas
            .iter()
            .map(|area| area.to_ascii_lowercase())
            .filter(|area| !area.is_empty())
            .collect();
        if areas.is_empty() {
            return 0.0;
        }

        let matched = session
            .criteria
            .iter()
            .filter(|criterion| {
                let haystack =
                    format!("{} {}", criterion.name, criterion.description).to_ascii_lowercase();
                areas.iter().any(|area| haystack.contains(area))
            })
            .count();
        matched as f64 / session.criteria.len() as f64
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

    /// Deterministic consensus: the median of the reviewers' ordinal ranks.
    ///
    /// A plain "most common recommendation" vote is not well-defined when
    /// there is a tie (e.g. two `Accept` and two `Reject`): breaking the tie
    /// by iterating a `HashMap` makes the outcome depend on hash iteration
    /// order, so the same set of reviews could yield a different consensus
    /// recommendation on different runs. The median is always well-defined,
    /// deterministic, and (unlike the mode) robust to a single outlier
    /// review.
    fn determine_consensus_recommendation(
        &self,
        recommendations: &[&ReviewRecommendation],
    ) -> ReviewRecommendation {
        if recommendations.is_empty() {
            return ReviewRecommendation::BorderlineReject;
        }

        let mut ranks: Vec<u8> = recommendations
            .iter()
            .map(|rec| Self::recommendation_rank(rec))
            .collect();
        ranks.sort_unstable();

        let mid = ranks.len() / 2;
        let median_rank = if ranks.len().is_multiple_of(2) {
            // Even count: average the two middle ranks, rounding toward the
            // more critical (reject) side on an exact half -- an "err on
            // the side of caution" convention that is itself deterministic.
            let lower = u16::from(ranks[mid - 1]);
            let upper = u16::from(ranks[mid]);
            (lower + upper).div_ceil(2) as u8
        } else {
            ranks[mid]
        };

        Self::recommendation_from_rank(median_rank)
    }

    /// Ordinal rank of a recommendation from most (0) to least (7)
    /// favorable, used to compute a deterministic median consensus.
    fn recommendation_rank(rec: &ReviewRecommendation) -> u8 {
        match rec {
            ReviewRecommendation::StrongAccept => 0,
            ReviewRecommendation::Accept => 1,
            ReviewRecommendation::WeakAccept => 2,
            ReviewRecommendation::BorderlineAccept => 3,
            ReviewRecommendation::BorderlineReject => 4,
            ReviewRecommendation::WeakReject => 5,
            ReviewRecommendation::Reject => 6,
            ReviewRecommendation::StrongReject => 7,
        }
    }

    /// Inverse of [`Self::recommendation_rank`].
    fn recommendation_from_rank(rank: u8) -> ReviewRecommendation {
        match rank {
            0 => ReviewRecommendation::StrongAccept,
            1 => ReviewRecommendation::Accept,
            2 => ReviewRecommendation::WeakAccept,
            3 => ReviewRecommendation::BorderlineAccept,
            4 => ReviewRecommendation::BorderlineReject,
            5 => ReviewRecommendation::WeakReject,
            6 => ReviewRecommendation::Reject,
            _ => ReviewRecommendation::StrongReject,
        }
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

    fn make_reviewer(id: &str) -> Reviewer {
        Reviewer {
            id: id.to_string(),
            expertise_areas: vec!["optimization".to_string()],
            experience_level: ExperienceLevel::Senior,
            review_history: ReviewerHistory::default(),
            availability: ReviewerAvailability::default(),
            quality_metrics: ReviewerQualityMetrics::default(),
            preferences: ReviewerPreferences {
                preferred_paper_types: Vec::new(),
                avoid_paper_types: Vec::new(),
                max_review_length: None,
                anonymous_preference: true,
                notification_preferences: NotificationPreferences::default(),
            },
        }
    }

    fn make_review(reviewer_id: &str, recommendation: ReviewRecommendation) -> PeerReview {
        PeerReview {
            id: uuid::Uuid::new_v4().to_string(),
            reviewer_id: reviewer_id.to_string(),
            recommendation,
            criterion_scores: HashMap::new(),
            overall_score: 3.0,
            confidence: 0.5,
            written_review: WrittenReview {
                summary: String::new(),
                strengths: Vec::new(),
                weaknesses: Vec::new(),
                detailed_comments: String::new(),
                questions: Vec::new(),
                minor_issues: Vec::new(),
                suggestions: Vec::new(),
                committee_comments: None,
            },
            status: ReviewStatus::Submitted,
            submitted_at: Some(Utc::now()),
            time_spent_minutes: Some(30),
        }
    }

    // Regression test for F73: `ln(0)` is `-inf`, and a review with no
    // written content at all (a realistic, valid input -- e.g. a
    // placeholder or a reviewer who only filled in scores) has
    // `review_length == 0`, which used to poison the entire quality score
    // to `-inf` instead of a valid score in `[0, 1]`.
    #[test]
    fn test_calculate_review_quality_handles_empty_review() {
        let system = PeerReviewSystem::new();
        let review = make_review("reviewer1", ReviewRecommendation::BorderlineAccept);

        let quality = system.calculate_review_quality(&review);

        assert!(
            quality.is_finite(),
            "quality score must be finite, got {quality}"
        );
        assert!(
            (0.0..=1.0).contains(&quality),
            "quality score must be in [0, 1], got {quality}"
        );
    }

    // Regression test for F74: consensus used to be "most frequent
    // recommendation, ties broken by HashMap iteration order" -- so a tied
    // vote could yield a different result on different runs for the exact
    // same input. It must now be the deterministic median, independent of
    // the order recommendations are supplied in.
    #[test]
    fn test_determine_consensus_recommendation_is_deterministic_median() {
        let system = PeerReviewSystem::new();

        // Tied 1-1 vote between Accept and Reject: the median of ranks
        // [1, 6] is rank 4 (BorderlineReject), not an order-dependent pick
        // of either tied recommendation.
        let accept = ReviewRecommendation::Accept;
        let reject = ReviewRecommendation::Reject;
        let order_a = system.determine_consensus_recommendation(&[&accept, &reject]);
        let order_b = system.determine_consensus_recommendation(&[&reject, &accept]);
        assert_eq!(order_a, ReviewRecommendation::BorderlineReject);
        assert_eq!(order_a, order_b, "consensus must not depend on input order");

        // A single outlier must not dominate the median the way it would a
        // naive average: [StrongAccept, Accept, Reject] medians to Accept.
        let strong_accept = ReviewRecommendation::StrongAccept;
        let median = system.determine_consensus_recommendation(&[&reject, &strong_accept, &accept]);
        assert_eq!(median, ReviewRecommendation::Accept);
    }

    // Regression test for F75: `assign_reviewers` never updated
    // `reviewer.availability.current_load`, so the load-based capacity
    // check in `get_available_reviewers` never actually reflected real
    // assignments; and neither `assign_reviewers` nor `submit_review`
    // guarded against the same reviewer being attached to / submitting for
    // one session twice.
    #[test]
    fn test_assign_reviewers_tracks_load_and_rejects_duplicates() {
        let mut system = PeerReviewSystem::new();
        system
            .reviewers
            .insert("reviewer1".to_string(), make_reviewer("reviewer1"));

        let deadline = Utc::now() + chrono::Duration::days(14);
        let session_id =
            system.create_review_session("paper1", ReviewType::DoubleBlind, vec![], deadline);

        // Duplicate ID within a single call must only produce one assignment.
        let ids = system
            .assign_reviewers(
                &session_id,
                &["reviewer1".to_string(), "reviewer1".to_string()],
                AssignmentMethod::Manual,
            )
            .expect("assignment should succeed");
        assert_eq!(ids.len(), 1);
        assert_eq!(system.reviewers["reviewer1"].availability.current_load, 1);

        // A second call for the same (session, reviewer) must be a no-op.
        let ids_again = system
            .assign_reviewers(
                &session_id,
                &["reviewer1".to_string()],
                AssignmentMethod::Manual,
            )
            .expect("assignment should succeed");
        assert!(ids_again.is_empty());
        assert_eq!(system.reviewers["reviewer1"].availability.current_load, 1);
        assert_eq!(system.calculate_reviewer_workload("reviewer1"), 1);

        // Submitting frees up the reviewer's active load...
        let review = make_review("reviewer1", ReviewRecommendation::Accept);
        system
            .submit_review(&session_id, "reviewer1", review)
            .expect("first submission should succeed");
        assert_eq!(system.reviewers["reviewer1"].availability.current_load, 0);

        // ...but a second submission for the same session must be rejected.
        let duplicate_review = make_review("reviewer1", ReviewRecommendation::Reject);
        let result = system.submit_review(&session_id, "reviewer1", duplicate_review);
        assert!(result.is_err(), "duplicate submission must be rejected");
        assert_eq!(
            system.sessions[&session_id].reviews.len(),
            1,
            "duplicate submission must not be recorded"
        );
    }
}
