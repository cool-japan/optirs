// Academic Research Collaboration Framework
//
// This module provides comprehensive tools for academic researchers to collaborate,
// reproduce experiments, track results, and generate publications using SciRS2 optimizers.
//
// # Features
//
// - **Experiment Tracking**: Comprehensive tracking of experimental configurations and results
// - **Reproducibility**: Tools to ensure experiments can be reproduced exactly
// - **Publication Generation**: Automated LaTeX/Markdown report generation
// - **Citation Management**: BibTeX integration and automated citations
// - **Benchmark Suite**: Standardized academic benchmarks
// - **Collaboration Tools**: Multi-researcher project management
// - **Conference Integration**: Templates and tools for academic conferences
// - **Peer Review Support**: Anonymous review and feedback systems
//
// # Examples
//
// ## Setting up a Research Project
//
// ```rust
// use optirs_core::research::*;
//
// let project = ResearchProject::new("Neural Optimizer Comparison")
//     .add_researcher("Dr. Alice Smith", "alice@university.edu")
//     .add_researcher("Prof. Bob Johnson", "bob@institute.org")
//     .set_research_area(ResearchArea::MachineLearning)
//     .set_funding_source("NSF Grant #12345")
//     .enable_reproducibility_tracking(true);
//
// // Setup experiment tracking
// let experiment = Experiment::new("Adam vs SGD Comparison")
//     .set_hypothesis("Adam converges faster than SGD on deep networks")
//     .add_optimizer_config("adam", adam_config)
//     .add_optimizer_config("sgd", sgd_config)
//     .set_dataset("CIFAR-10")
//     .set_metrics(vec!["accuracy", "convergence_rate", "training_time"]);
// ```

pub mod benchmarks;
pub mod citations;
pub mod collaboration;
pub mod conferences;
pub mod datasets;
pub mod experiments;
pub mod funding;
pub mod peer_review;
pub mod publications;
pub mod reproducibility;

// Re-exports for convenience - using selective exports to avoid conflicts
pub use benchmarks::*;
pub use collaboration::*;
pub use datasets::*;
pub use funding::*;

// Selective re-exports for experiments to avoid conflicts
pub use experiments::{
    CpuInfo, DataType, DatasetInfo, DatasetStatistics, EarlyStoppingConfig, Experiment,
    ExperimentConfig, ExperimentMetadata, ExperimentNote, ExperimentResult, ExperimentRunner,
    ExperimentStatus, ExperimentTimeline, GpuInfo, HardwareConfig as ExperimentHardwareConfig,
    MemoryAllocationStrategy, MemoryConfig, NoteType, OptimizationMode, ParallelConfig,
    PreprocessingStep, ReproducibilityChecklist, ReproducibilityInfo, ResourceMonitor,
    ResourceUsage, RunStatus, SystemInfo as ExperimentSystemInfo, TrainingHistory,
};

// Selective re-exports for reproducibility to avoid conflicts
pub use reproducibility::{
    CpuSpec, Dependency, Difference, DifferenceCategory, EnvironmentSnapshot, GpuSpec,
    HardwareConfig as ReproducibilityHardwareConfig, IssueSeverity, IssueType, MemorySpec,
    ReproducibilityChecklist as ReproducibilityChecklistType, ReproducibilityConfig,
    ReproducibilityIssue, ReproducibilityManager, ReproducibilityReport, ReproducibilityStorage,
    SimilarityMetrics, StorageSpec, SystemInfo as ReproducibilitySystemInfo, VerificationResult,
    VerificationStatus,
};

// Selective re-exports to avoid conflicts
// Citations module exports
pub use citations::{
    Author as CitationAuthor, BibliographyFormat, Citation, CitationManager,
    CitationStyle as CitationStyleType, DateFormat, FormattingRules, InTextFormat, NameFormat,
    PublicationType as CitationPublicationType, PunctuationRules, SortDirection, SortField,
    SortingRules, TitleFormat,
};

// Publications module exports
pub use publications::{
    Affiliation, Author as PublicationAuthor, AuthorPosition, BibTeXEntry, Bibliography,
    CitationStyle as PublicationCitationStyle, Decision, Figure, FigureType, ManuscriptSection,
    Publication, PublicationStatus, PublicationType as PublicationTypeEnum,
    Review as PublicationReview, ReviewRecommendation as PublicationReviewRecommendation,
    ReviewerInfo, SectionType, SubmissionRecord, SubmissionStatus, Table, Venue, VenueType,
};

// Peer review module exports
pub use peer_review::{
    MetaReview, PeerReview, PeerReviewSystem, ReviewAssignment, ReviewCriterion, ReviewDiscussion,
    ReviewQualityAssessment, ReviewQualityMetric, ReviewRecommendation as PeerReviewRecommendation,
    ReviewSession, ReviewSessionStatus, ReviewStatus, ReviewType, Reviewer, ReviewerAvailability,
    ReviewerHistory, ReviewerPreferences, ReviewerQualityMetrics, WrittenReview,
};

// Conferences module exports
pub use conferences::{
    Conference, ConferenceDates, ConferenceManager, DeadlineType, Review as ConferenceReview,
    ReviewFormat, ReviewProcess, ReviewRecommendation as ConferenceReviewRecommendation,
    Submission, SubmissionStatus as ConferenceSubmissionStatus,
};

use crate::error::{OptimError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Main research project coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchProject {
    /// Project name
    pub name: String,
    /// Project description
    pub description: String,
    /// Research area classification
    pub research_area: ResearchArea,
    /// Project status
    pub status: ProjectStatus,
    /// List of researchers
    pub researchers: Vec<Researcher>,
    /// Funding information
    pub funding: Option<FundingInfo>,
    /// Project timeline
    pub timeline: ProjectTimeline,
    /// Experiments in this project
    pub experiments: Vec<Experiment>,
    /// Publications related to this project
    pub publications: Vec<Publication>,
    /// Project settings
    pub settings: ProjectSettings,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
}

/// Research area classifications
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResearchArea {
    /// Machine Learning
    MachineLearning,
    /// Deep Learning
    DeepLearning,
    /// Optimization Theory
    OptimizationTheory,
    /// Computer Vision
    ComputerVision,
    /// Natural Language Processing
    NaturalLanguageProcessing,
    /// Reinforcement Learning
    ReinforcementLearning,
    /// Scientific Computing
    ScientificComputing,
    /// High Performance Computing
    HighPerformanceComputing,
    /// Distributed Systems
    DistributedSystems,
    /// Quantum Computing
    QuantumComputing,
    /// Interdisciplinary
    Interdisciplinary,
    /// Other with description
    Other(String),
}

/// Project status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProjectStatus {
    /// Planning phase
    Planning,
    /// Active research
    Active,
    /// Data collection
    DataCollection,
    /// Analysis phase
    Analysis,
    /// Writing phase
    Writing,
    /// Under review
    UnderReview,
    /// Published
    Published,
    /// Completed
    Completed,
    /// On hold
    OnHold,
    /// Cancelled
    Cancelled,
}

/// Researcher information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Researcher {
    /// Full name
    pub name: String,
    /// Email address
    pub email: String,
    /// Institution/affiliation
    pub affiliation: String,
    /// ORCID identifier
    pub orcid: Option<String>,
    /// Role in the project
    pub role: ResearcherRole,
    /// Expertise areas
    pub expertise: Vec<String>,
    /// Contact information
    pub contact_info: ContactInfo,
}

/// Researcher roles
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResearcherRole {
    /// Principal Investigator
    PrincipalInvestigator,
    /// Co-Principal Investigator
    CoPrincipalInvestigator,
    /// Senior Researcher
    SeniorResearcher,
    /// Postdoc
    Postdoc,
    /// PhD Student
    PhDStudent,
    /// Masters Student
    MastersStudent,
    /// Undergraduate Student
    UndergraduateStudent,
    /// Research Assistant
    ResearchAssistant,
    /// Collaborator
    Collaborator,
    /// Advisor
    Advisor,
}

/// Contact information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactInfo {
    /// Phone number
    pub phone: Option<String>,
    /// Office address
    pub office_address: Option<String>,
    /// Mailing address
    pub mailing_address: Option<String>,
    /// Website/homepage
    pub website: Option<String>,
    /// Social media handles
    pub social_media: HashMap<String, String>,
}

/// Funding information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingInfo {
    /// Funding agency
    pub agency: String,
    /// Grant number
    pub grant_number: String,
    /// Grant title
    pub grant_title: String,
    /// Total amount
    pub amount: Option<f64>,
    /// Currency
    pub currency: String,
    /// Start date
    pub start_date: DateTime<Utc>,
    /// End date
    pub end_date: DateTime<Utc>,
    /// Award date
    pub award_date: Option<DateTime<Utc>>,
}

/// Project timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectTimeline {
    /// Project start date
    pub start_date: DateTime<Utc>,
    /// Planned end date
    pub planned_end_date: DateTime<Utc>,
    /// Actual end date
    pub actual_end_date: Option<DateTime<Utc>>,
    /// Key milestones
    pub milestones: Vec<Milestone>,
    /// Progress updates
    pub progress_updates: Vec<ProgressUpdate>,
}

/// Project milestone
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    /// Milestone name
    pub name: String,
    /// Description
    pub description: String,
    /// Planned date
    pub planned_date: DateTime<Utc>,
    /// Actual completion date
    pub completed_date: Option<DateTime<Utc>>,
    /// Status
    pub status: MilestoneStatus,
    /// Deliverables
    pub deliverables: Vec<String>,
}

/// Milestone status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MilestoneStatus {
    /// Not started
    NotStarted,
    /// In progress
    InProgress,
    /// Completed
    Completed,
    /// Delayed
    Delayed,
    /// Cancelled
    Cancelled,
}

/// Progress update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressUpdate {
    /// Update timestamp
    pub timestamp: DateTime<Utc>,
    /// Update author
    pub author: String,
    /// Progress summary
    pub summary: String,
    /// Detailed description
    pub description: String,
    /// Percentage completion
    pub completion_percentage: f64,
    /// Next steps
    pub next_steps: Vec<String>,
    /// Challenges encountered
    pub challenges: Vec<String>,
}

/// Project settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectSettings {
    /// Enable reproducibility tracking
    pub enable_reproducibility: bool,
    /// Enable automatic backups
    pub enable_backups: bool,
    /// Backup frequency (hours)
    pub backup_frequency_hours: u32,
    /// Data retention policy (days)
    pub data_retention_days: u32,
    /// Privacy settings
    pub privacy_settings: PrivacySettings,
    /// Collaboration settings
    pub collaboration_settings: CollaborationSettings,
    /// Publication settings
    pub publication_settings: PublicationSettings,
}

/// Privacy settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacySettings {
    /// Project visibility
    pub visibility: ProjectVisibility,
    /// Data sharing policy
    pub data_sharing: DataSharingPolicy,
    /// Enable anonymous access
    pub allow_anonymous_access: bool,
    /// Embargo period (days)
    pub embargo_period_days: Option<u32>,
}

/// Project visibility
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProjectVisibility {
    /// Public - visible to everyone
    Public,
    /// Internal - visible to institution members
    Internal,
    /// Private - visible to project members only
    Private,
    /// Restricted - invitation only
    Restricted,
}

/// Data sharing policy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataSharingPolicy {
    /// Open data - freely available
    Open,
    /// On request - available upon request
    OnRequest,
    /// Restricted - limited sharing
    Restricted,
    /// No sharing - private data only
    NoSharing,
}

/// Collaboration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationSettings {
    /// Enable real-time collaboration
    pub enable_realtime_collaboration: bool,
    /// Enable version control
    pub enable_version_control: bool,
    /// Enable peer review
    pub enable_peer_review: bool,
    /// Enable discussion forums
    pub enable_discussions: bool,
    /// Maximum collaborators
    pub max_collaborators: Option<u32>,
}

/// Publication settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicationSettings {
    /// Default publication format
    pub default_format: PublicationFormat,
    /// Enable automatic citations
    pub enable_auto_citations: bool,
    /// Citation style
    pub citation_style: CitationStyle,
    /// Enable preprint submission
    pub enable_preprints: bool,
    /// Target conferences/journals
    pub target_venues: Vec<String>,
}

/// Publication formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PublicationFormat {
    /// LaTeX
    LaTeX,
    /// Markdown
    Markdown,
    /// HTML
    Html,
    /// PDF
    Pdf,
    /// Word Document
    Word,
    /// Jupyter Notebook
    Notebook,
}

/// Citation styles
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CitationStyle {
    /// APA style
    APA,
    /// IEEE style
    IEEE,
    /// ACM style
    ACM,
    /// Nature style
    Nature,
    /// Science style
    Science,
    /// Custom BibTeX style
    Custom(String),
}

/// Research project manager
#[derive(Debug)]
pub struct ResearchProjectManager {
    /// Active projects
    projects: HashMap<String, ResearchProject>,
    /// Project storage directory
    storage_dir: PathBuf,
    /// Manager settings
    settings: ManagerSettings,
}

/// Manager settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerSettings {
    /// Default project template
    pub default_template: Option<String>,
    /// Auto-save frequency (minutes)
    pub auto_save_frequency: u32,
    /// Enable cloud backup
    pub enable_cloud_backup: bool,
    /// Cloud storage provider
    pub cloud_provider: Option<String>,
    /// Notification settings
    pub notifications: NotificationSettings,
}

/// Notification settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    /// Email notifications enabled
    pub email_enabled: bool,
    /// Slack notifications enabled
    pub slack_enabled: bool,
    /// Webhook URL for notifications
    pub webhook_url: Option<String>,
    /// Notification frequency
    pub frequency: NotificationFrequency,
}

/// Notification frequency
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NotificationFrequency {
    /// Real-time notifications
    Realtime,
    /// Daily digest
    Daily,
    /// Weekly digest
    Weekly,
    /// Monthly digest
    Monthly,
    /// On important events only
    Important,
    /// Disabled
    Disabled,
}

impl ResearchProject {
    /// Create a new research project
    pub fn new(name: &str) -> Self {
        let now = Utc::now();
        Self {
            name: name.to_string(),
            description: String::new(),
            research_area: ResearchArea::MachineLearning,
            status: ProjectStatus::Planning,
            researchers: Vec::new(),
            funding: None,
            timeline: ProjectTimeline {
                start_date: now,
                planned_end_date: now + chrono::Duration::days(365),
                actual_end_date: None,
                milestones: Vec::new(),
                progress_updates: Vec::new(),
            },
            experiments: Vec::new(),
            publications: Vec::new(),
            settings: ProjectSettings::default(),
            created_at: now,
            modified_at: now,
        }
    }

    /// Set project description
    pub fn description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self.modified_at = Utc::now();
        self
    }

    /// Set research area
    pub fn research_area(mut self, area: ResearchArea) -> Self {
        self.research_area = area;
        self.modified_at = Utc::now();
        self
    }

    /// Add a researcher to the project
    pub fn add_researcher(mut self, name: &str, email: &str) -> Self {
        let researcher = Researcher {
            name: name.to_string(),
            email: email.to_string(),
            affiliation: String::new(),
            orcid: None,
            role: ResearcherRole::Collaborator,
            expertise: Vec::new(),
            contact_info: ContactInfo::default(),
        };
        self.researchers.push(researcher);
        self.modified_at = Utc::now();
        self
    }

    /// Set funding information
    pub fn funding(mut self, funding: FundingInfo) -> Self {
        self.funding = Some(funding);
        self.modified_at = Utc::now();
        self
    }

    /// Enable reproducibility tracking
    pub fn enable_reproducibility(mut self, enabled: bool) -> Self {
        self.settings.enable_reproducibility = enabled;
        self.modified_at = Utc::now();
        self
    }

    /// Add an experiment to the project
    pub fn add_experiment(&mut self, experiment: Experiment) -> Result<()> {
        self.experiments.push(experiment);
        self.modified_at = Utc::now();
        Ok(())
    }

    /// Add a publication to the project
    pub fn add_publication(&mut self, publication: Publication) -> Result<()> {
        self.publications.push(publication);
        self.modified_at = Utc::now();
        Ok(())
    }

    /// Generate project summary report
    pub fn generate_summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("# {}\n\n", self.name));
        report.push_str(&format!("**Description**: {}\n\n", self.description));
        report.push_str(&format!("**Research Area**: {:?}\n\n", self.research_area));
        report.push_str(&format!("**Status**: {:?}\n\n", self.status));

        report.push_str("## Researchers\n\n");
        for researcher in &self.researchers {
            report.push_str(&format!(
                "- {} ({}) - {:?}\n",
                researcher.name, researcher.email, researcher.role
            ));
        }

        if let Some(funding) = &self.funding {
            report.push_str(&format!("\n## Funding\n\n"));
            report.push_str(&format!("**Agency**: {}\n", funding.agency));
            report.push_str(&format!("**Grant**: {}\n", funding.grant_number));
            if let Some(amount) = funding.amount {
                report.push_str(&format!("**Amount**: {:.2} {}\n", amount, funding.currency));
            }
        }

        report.push_str(&format!("\n## Progress\n\n"));
        report.push_str(&format!("**Experiments**: {}\n", self.experiments.len()));
        report.push_str(&format!("**Publications**: {}\n", self.publications.len()));
        report.push_str(&format!(
            "**Milestones**: {}\n",
            self.timeline.milestones.len()
        ));

        report
    }
}

impl Default for ContactInfo {
    fn default() -> Self {
        Self {
            phone: None,
            office_address: None,
            mailing_address: None,
            website: None,
            social_media: HashMap::new(),
        }
    }
}

impl Default for ProjectSettings {
    fn default() -> Self {
        Self {
            enable_reproducibility: true,
            enable_backups: true,
            backup_frequency_hours: 24,
            data_retention_days: 1095, // 3 years
            privacy_settings: PrivacySettings::default(),
            collaboration_settings: CollaborationSettings::default(),
            publication_settings: PublicationSettings::default(),
        }
    }
}

impl Default for PrivacySettings {
    fn default() -> Self {
        Self {
            visibility: ProjectVisibility::Private,
            data_sharing: DataSharingPolicy::OnRequest,
            allow_anonymous_access: false,
            embargo_period_days: Some(365),
        }
    }
}

impl Default for CollaborationSettings {
    fn default() -> Self {
        Self {
            enable_realtime_collaboration: true,
            enable_version_control: true,
            enable_peer_review: true,
            enable_discussions: true,
            max_collaborators: Some(20),
        }
    }
}

impl Default for PublicationSettings {
    fn default() -> Self {
        Self {
            default_format: PublicationFormat::LaTeX,
            enable_auto_citations: true,
            citation_style: CitationStyle::IEEE,
            enable_preprints: true,
            target_venues: Vec::new(),
        }
    }
}

impl ResearchProjectManager {
    /// Create a new research project manager
    pub fn new(storage_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&storage_dir)?;

        Ok(Self {
            projects: HashMap::new(),
            storage_dir,
            settings: ManagerSettings::default(),
        })
    }

    /// Create a new research project
    pub fn create_project(&mut self, name: &str) -> Result<&mut ResearchProject> {
        if self.projects.contains_key(name) {
            return Err(OptimError::InvalidConfig(format!(
                "Project '{}' already exists",
                name
            )));
        }

        let project = ResearchProject::new(name);
        self.projects.insert(name.to_string(), project);
        self.save_project(name)?;

        Ok(self.projects.get_mut(name).unwrap())
    }

    /// Load an existing project
    pub fn load_project(&mut self, name: &str) -> Result<&mut ResearchProject> {
        if !self.projects.contains_key(name) {
            let project_file = self.storage_dir.join(format!("{}.json", name));
            if project_file.exists() {
                let content = std::fs::read_to_string(&project_file)?;
                let project: ResearchProject = serde_json::from_str(&content)?;
                self.projects.insert(name.to_string(), project);
            } else {
                return Err(OptimError::InvalidConfig(format!(
                    "Project '{}' not found",
                    name
                )));
            }
        }

        Ok(self.projects.get_mut(name).unwrap())
    }

    /// Save a project to disk
    pub fn save_project(&self, name: &str) -> Result<()> {
        if let Some(project) = self.projects.get(name) {
            let project_file = self.storage_dir.join(format!("{}.json", name));
            let content = serde_json::to_string_pretty(project)?;
            std::fs::write(&project_file, content)?;
        }

        Ok(())
    }

    /// List all projects
    pub fn list_projects(&self) -> Vec<&str> {
        self.projects.keys().map(|s| s.as_str()).collect()
    }

    /// Delete a project
    pub fn delete_project(&mut self, name: &str) -> Result<()> {
        self.projects.remove(name);
        let project_file = self.storage_dir.join(format!("{}.json", name));
        if project_file.exists() {
            std::fs::remove_file(&project_file)?;
        }

        Ok(())
    }
}

impl Default for ManagerSettings {
    fn default() -> Self {
        Self {
            default_template: None,
            auto_save_frequency: 15, // 15 minutes
            enable_cloud_backup: false,
            cloud_provider: None,
            notifications: NotificationSettings::default(),
        }
    }
}

impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            email_enabled: true,
            slack_enabled: false,
            webhook_url: None,
            frequency: NotificationFrequency::Daily,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_research_project_creation() {
        let project = ResearchProject::new("Test Project")
            .description("A test research project")
            .research_area(ResearchArea::MachineLearning)
            .add_researcher("Dr. Test", "test@example.com")
            .enable_reproducibility(true);

        assert_eq!(project.name, "Test Project");
        assert_eq!(project.description, "A test research project");
        assert_eq!(project.research_area, ResearchArea::MachineLearning);
        assert_eq!(project.researchers.len(), 1);
        assert!(project.settings.enable_reproducibility);
    }

    #[test]
    fn test_project_manager() {
        let temp_dir = tempdir().unwrap();
        let mut manager = ResearchProjectManager::new(temp_dir.path().to_path_buf()).unwrap();

        let project = manager.create_project("Test Project").unwrap();
        project.description = "Test description".to_string();

        manager.save_project("Test Project").unwrap();

        let projects = manager.list_projects();
        assert!(projects.contains(&"Test Project"));

        manager.delete_project("Test Project").unwrap();
        let projects = manager.list_projects();
        assert!(!projects.contains(&"Test Project"));
    }
}
