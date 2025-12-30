// Collaboration tools for multi-researcher projects
//
// This module provides tools for managing collaborative research projects,
// including real-time editing, version control, communication, and task management.

use crate::error::{OptimError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Collaborative workspace for research projects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeWorkspace {
    /// Workspace identifier
    pub id: String,
    /// Workspace name
    pub name: String,
    /// Project members
    pub members: Vec<ProjectMember>,
    /// Shared documents
    pub documents: Vec<SharedDocument>,
    /// Communication channels
    pub channels: Vec<CommunicationChannel>,
    /// Task assignments
    pub tasks: Vec<Task>,
    /// Version control information
    pub version_control: VersionControl,
    /// Workspace settings
    pub settings: WorkspaceSettings,
    /// Access control
    pub access_control: AccessControl,
    /// Activity log
    pub activity_log: Vec<Activity>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
}

/// Project member information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMember {
    /// Member identifier
    pub id: String,
    /// User information
    pub user: UserInfo,
    /// Member role
    pub role: MemberRole,
    /// Permissions
    pub permissions: Vec<Permission>,
    /// Join date
    pub joined_at: DateTime<Utc>,
    /// Last active timestamp
    pub last_active: DateTime<Utc>,
    /// Member status
    pub status: MemberStatus,
    /// Contribution statistics
    pub contributions: ContributionStats,
}

/// User information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    /// Full name
    pub name: String,
    /// Email address
    pub email: String,
    /// Institution
    pub institution: String,
    /// Profile picture URL
    pub avatar_url: Option<String>,
    /// Timezone
    pub timezone: String,
    /// Preferred language
    pub language: String,
    /// Research interests
    pub research_interests: Vec<String>,
}

/// Member roles
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemberRole {
    /// Project owner
    Owner,
    /// Project administrator
    Admin,
    /// Principal investigator
    PrincipalInvestigator,
    /// Senior researcher
    SeniorResearcher,
    /// Researcher
    Researcher,
    /// PhD student
    PhDStudent,
    /// Masters student
    MastersStudent,
    /// Research assistant
    ResearchAssistant,
    /// Collaborator
    Collaborator,
    /// Guest
    Guest,
    /// Observer (read-only)
    Observer,
}

/// Member permissions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Permission {
    /// Read project data
    Read,
    /// Write/edit project data
    Write,
    /// Delete project data
    Delete,
    /// Manage members
    ManageMembers,
    /// Manage permissions
    ManagePermissions,
    /// Manage settings
    ManageSettings,
    /// Create experiments
    CreateExperiments,
    /// Run experiments
    RunExperiments,
    /// Publish results
    PublishResults,
    /// Access sensitive data
    AccessSensitiveData,
}

/// Member status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemberStatus {
    /// Active member
    Active,
    /// Inactive member
    Inactive,
    /// On leave
    OnLeave,
    /// Suspended
    Suspended,
    /// Former member
    Former,
}

/// Contribution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionStats {
    /// Number of experiments created
    pub experiments_created: usize,
    /// Number of experiments run
    pub experiments_run: usize,
    /// Lines of code contributed
    pub lines_of_code: usize,
    /// Documents authored
    pub documents_authored: usize,
    /// Comments/discussions posted
    pub comments_posted: usize,
    /// Reviews conducted
    pub reviews_conducted: usize,
    /// Total contribution score
    pub contribution_score: f64,
}

/// Shared document in the workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedDocument {
    /// Document identifier
    pub id: String,
    /// Document name
    pub name: String,
    /// Document type
    pub document_type: DocumentType,
    /// Document content
    pub content: String,
    /// Document owner
    pub owner_id: String,
    /// Collaborators with edit access
    pub collaborators: Vec<String>,
    /// Document version
    pub version: u32,
    /// Version history
    pub version_history: Vec<DocumentVersion>,
    /// Access permissions
    pub access_permissions: DocumentPermissions,
    /// Document metadata
    pub metadata: DocumentMetadata,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
}

/// Document types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DocumentType {
    /// Research paper/manuscript
    Manuscript,
    /// Experiment notes
    ExperimentNotes,
    /// Meeting notes
    MeetingNotes,
    /// Literature review
    LiteratureReview,
    /// Research proposal
    ResearchProposal,
    /// Data analysis
    DataAnalysis,
    /// Code documentation
    CodeDocumentation,
    /// Presentation
    Presentation,
    /// Other document
    Other(String),
}

/// Document version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentVersion {
    /// Version number
    pub version: u32,
    /// Version author
    pub author_id: String,
    /// Version content
    pub content: String,
    /// Change summary
    pub change_summary: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Content hash for integrity
    pub content_hash: String,
}

/// Document permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentPermissions {
    /// Public visibility
    pub public: bool,
    /// Read access
    pub read_access: Vec<String>,
    /// Write access
    pub write_access: Vec<String>,
    /// Admin access
    pub admin_access: Vec<String>,
}

/// Document metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Document tags
    pub tags: Vec<String>,
    /// Word count
    pub word_count: usize,
    /// Character count
    pub character_count: usize,
    /// Number of collaborators
    pub collaborator_count: usize,
    /// Number of versions
    pub version_count: usize,
    /// Last editor
    pub last_editor_id: String,
}

/// Communication channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationChannel {
    /// Channel identifier
    pub id: String,
    /// Channel name
    pub name: String,
    /// Channel description
    pub description: String,
    /// Channel type
    pub channel_type: ChannelType,
    /// Channel members
    pub members: Vec<String>,
    /// Messages in the channel
    pub messages: Vec<Message>,
    /// Channel settings
    pub settings: ChannelSettings,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Channel types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChannelType {
    /// General discussion
    General,
    /// Experiment discussion
    Experiments,
    /// Paper writing
    PaperWriting,
    /// Code review
    CodeReview,
    /// Announcements
    Announcements,
    /// Random/off-topic
    Random,
    /// Private channel
    Private,
    /// Direct message
    DirectMessage,
}

/// Message in a communication channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message identifier
    pub id: String,
    /// Message author
    pub author_id: String,
    /// Message content
    pub content: String,
    /// Message type
    pub message_type: MessageType,
    /// Attachments
    pub attachments: Vec<Attachment>,
    /// Replies to this message
    pub replies: Vec<Message>,
    /// Reactions
    pub reactions: Vec<Reaction>,
    /// Mentions
    pub mentions: Vec<String>,
    /// Thread ID (if part of a thread)
    pub thread_id: Option<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Edit history
    pub edit_history: Vec<MessageEdit>,
}

/// Message types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageType {
    /// Regular text message
    Text,
    /// Code snippet
    Code,
    /// File attachment
    File,
    /// System message
    System,
    /// Experiment result
    ExperimentResult,
    /// Task assignment
    TaskAssignment,
    /// Meeting invitation
    MeetingInvitation,
}

/// File attachment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    /// File name
    pub filename: String,
    /// File size in bytes
    pub size: usize,
    /// MIME type
    pub mime_type: String,
    /// File path or URL
    pub file_path: String,
    /// File hash for integrity
    pub file_hash: String,
    /// Upload timestamp
    pub uploaded_at: DateTime<Utc>,
}

/// Message reaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reaction {
    /// Emoji or reaction type
    pub emoji: String,
    /// Users who reacted
    pub users: Vec<String>,
    /// Reaction count
    pub count: usize,
}

/// Message edit history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEdit {
    /// Original content
    pub original_content: String,
    /// Edit timestamp
    pub edited_at: DateTime<Utc>,
    /// Edit reason
    pub edit_reason: Option<String>,
}

/// Channel settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelSettings {
    /// Notifications enabled
    pub notifications: bool,
    /// Archive old messages
    pub auto_archive: bool,
    /// Archive threshold (days)
    pub archive_after_days: u32,
    /// Allow external invites
    pub allow_external_invites: bool,
    /// Moderation settings
    pub moderation: ModerationSettings,
}

/// Moderation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationSettings {
    /// Require approval for new messages
    pub require_approval: bool,
    /// Auto-delete inappropriate content
    pub auto_delete_inappropriate: bool,
    /// Spam filtering enabled
    pub spam_filtering: bool,
    /// Moderators
    pub moderators: Vec<String>,
}

/// Task in the collaborative workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Task identifier
    pub id: String,
    /// Task title
    pub title: String,
    /// Task description
    pub description: String,
    /// Task type
    pub task_type: TaskType,
    /// Task status
    pub status: TaskStatus,
    /// Task priority
    pub priority: TaskPriority,
    /// Assigned to
    pub assigned_to: Vec<String>,
    /// Created by
    pub created_by: String,
    /// Due date
    pub due_date: Option<DateTime<Utc>>,
    /// Estimated effort (hours)
    pub estimated_hours: Option<f64>,
    /// Actual effort (hours)
    pub actual_hours: Option<f64>,
    /// Task dependencies
    pub dependencies: Vec<String>,
    /// Subtasks
    pub subtasks: Vec<Task>,
    /// Comments
    pub comments: Vec<TaskComment>,
    /// Attachments
    pub attachments: Vec<Attachment>,
    /// Labels/tags
    pub labels: Vec<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Completion timestamp
    pub completed_at: Option<DateTime<Utc>>,
}

/// Task types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskType {
    /// Experiment design
    ExperimentDesign,
    /// Data collection
    DataCollection,
    /// Data analysis
    DataAnalysis,
    /// Code development
    CodeDevelopment,
    /// Documentation
    Documentation,
    /// Literature review
    LiteratureReview,
    /// Paper writing
    PaperWriting,
    /// Review/feedback
    Review,
    /// Meeting
    Meeting,
    /// Administrative
    Administrative,
    /// Other
    Other(String),
}

/// Task status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task not started
    NotStarted,
    /// Task in progress
    InProgress,
    /// Task on hold
    OnHold,
    /// Task completed
    Completed,
    /// Task cancelled
    Cancelled,
    /// Task needs review
    NeedsReview,
    /// Task approved
    Approved,
}

/// Task priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskPriority {
    /// Critical priority
    Critical,
    /// High priority
    High,
    /// Medium priority
    Medium,
    /// Low priority
    Low,
}

/// Task comment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskComment {
    /// Comment ID
    pub id: String,
    /// Comment author
    pub author_id: String,
    /// Comment content
    pub content: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Version control system for the workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionControl {
    /// Repository URL
    pub repository_url: Option<String>,
    /// Current branch
    pub current_branch: String,
    /// Available branches
    pub branches: Vec<String>,
    /// Commit history
    pub commits: Vec<Commit>,
    /// Merge requests
    pub merge_requests: Vec<MergeRequest>,
    /// Version control settings
    pub settings: VersionControlSettings,
}

/// Git commit information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    /// Commit hash
    pub hash: String,
    /// Commit author
    pub author: String,
    /// Commit message
    pub message: String,
    /// Commit timestamp
    pub timestamp: DateTime<Utc>,
    /// Modified files
    pub modified_files: Vec<String>,
    /// Parent commits
    pub parents: Vec<String>,
}

/// Merge/Pull request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRequest {
    /// Request ID
    pub id: String,
    /// Request title
    pub title: String,
    /// Request description
    pub description: String,
    /// Source branch
    pub source_branch: String,
    /// Target branch
    pub target_branch: String,
    /// Request author
    pub author: String,
    /// Reviewers
    pub reviewers: Vec<String>,
    /// Request status
    pub status: MergeRequestStatus,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Merge timestamp
    pub merged_at: Option<DateTime<Utc>>,
}

/// Merge request status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MergeRequestStatus {
    /// Open for review
    Open,
    /// Under review
    UnderReview,
    /// Approved
    Approved,
    /// Merged
    Merged,
    /// Closed without merging
    Closed,
    /// Draft
    Draft,
}

/// Version control settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionControlSettings {
    /// Auto-commit enabled
    pub auto_commit: bool,
    /// Auto-commit frequency (minutes)
    pub auto_commit_frequency: u32,
    /// Require review for merges
    pub require_review: bool,
    /// Protected branches
    pub protected_branches: Vec<String>,
    /// Automatic backups
    pub automatic_backups: bool,
}

/// Workspace settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSettings {
    /// Workspace timezone
    pub timezone: String,
    /// Default language
    pub default_language: String,
    /// Collaboration settings
    pub collaboration: CollaborationSettings,
    /// Notification settings
    pub notifications: NotificationSettings,
    /// Integration settings
    pub integrations: IntegrationSettings,
}

/// Collaboration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationSettings {
    /// Real-time editing enabled
    pub real_time_editing: bool,
    /// Auto-save frequency (seconds)
    pub auto_save_frequency: u32,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolution,
    /// Maximum simultaneous editors
    pub max_simultaneous_editors: u32,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConflictResolution {
    /// Manual resolution required
    Manual,
    /// Last writer wins
    LastWriterWins,
    /// First writer wins
    FirstWriterWins,
    /// Merge changes
    Merge,
}

/// Notification settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    /// Email notifications
    pub email: bool,
    /// In-app notifications
    pub in_app: bool,
    /// Desktop notifications
    pub desktop: bool,
    /// Mobile push notifications
    pub mobile_push: bool,
    /// Notification frequency
    pub frequency: NotificationFrequency,
}

/// Notification frequency
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NotificationFrequency {
    /// Immediate notifications
    Immediate,
    /// Digest every hour
    Hourly,
    /// Daily digest
    Daily,
    /// Weekly digest
    Weekly,
    /// No notifications
    None,
}

/// Integration settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IntegrationSettings {
    /// Slack integration
    pub slack: Option<SlackIntegration>,
    /// Email integration
    pub email: Option<EmailIntegration>,
    /// Calendar integration
    pub calendar: Option<CalendarIntegration>,
    /// Cloud storage integration
    pub cloud_storage: Option<CloudStorageIntegration>,
}

/// Slack integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackIntegration {
    /// Webhook URL
    pub webhook_url: String,
    /// Default channel
    pub default_channel: String,
    /// Enable experiment notifications
    pub experiment_notifications: bool,
    /// Enable task notifications
    pub task_notifications: bool,
}

/// Email integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailIntegration {
    /// SMTP server
    pub smtp_server: String,
    /// SMTP port
    pub smtp_port: u16,
    /// Email address
    pub email_address: String,
    /// Authentication credentials
    pub auth_credentials: Option<String>,
}

/// Calendar integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarIntegration {
    /// Calendar provider
    pub provider: CalendarProvider,
    /// Calendar ID
    pub calendar_id: String,
    /// Sync meetings
    pub sync_meetings: bool,
    /// Sync deadlines
    pub sync_deadlines: bool,
}

/// Calendar providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CalendarProvider {
    /// Google Calendar
    Google,
    /// Outlook Calendar
    Outlook,
    /// Apple Calendar
    Apple,
    /// CalDAV
    CalDAV,
}

/// Cloud storage integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudStorageIntegration {
    /// Storage provider
    pub provider: CloudStorageProvider,
    /// Storage path
    pub storage_path: String,
    /// Auto-sync enabled
    pub auto_sync: bool,
    /// Sync frequency (minutes)
    pub sync_frequency: u32,
}

/// Cloud storage providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CloudStorageProvider {
    /// Google Drive
    GoogleDrive,
    /// Dropbox
    Dropbox,
    /// OneDrive
    OneDrive,
    /// Amazon S3
    AmazonS3,
    /// Custom provider
    Custom(String),
}

/// Access control for the workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControl {
    /// Access control lists
    pub acls: Vec<AccessControlEntry>,
    /// Default permissions for new members
    pub default_permissions: Vec<Permission>,
    /// Guest access allowed
    pub guest_access: bool,
    /// Public visibility
    pub public_visibility: bool,
    /// Invitation settings
    pub invitation_settings: InvitationSettings,
}

/// Access control entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlEntry {
    /// Principal (user, group, or role)
    pub principal: Principal,
    /// Granted permissions
    pub permissions: Vec<Permission>,
    /// Access expiration
    pub expires_at: Option<DateTime<Utc>>,
}

/// Principal (user, group, or role)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Principal {
    /// Individual user
    User(String),
    /// User group
    Group(String),
    /// Member role
    Role(MemberRole),
    /// Everyone
    Everyone,
}

/// Invitation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvitationSettings {
    /// Require approval for invitations
    pub require_approval: bool,
    /// Allow external invitations
    pub allow_external: bool,
    /// Invitation expiration (days)
    pub expiration_days: u32,
    /// Maximum invitations per user
    pub max_invitations_per_user: u32,
}

/// Activity log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Activity {
    /// Activity ID
    pub id: String,
    /// User who performed the activity
    pub user_id: String,
    /// Activity type
    pub activity_type: ActivityType,
    /// Activity description
    pub description: String,
    /// Affected resources
    pub resources: Vec<String>,
    /// Activity metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Activity types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActivityType {
    /// User joined workspace
    UserJoined,
    /// User left workspace
    UserLeft,
    /// Document created
    DocumentCreated,
    /// Document edited
    DocumentEdited,
    /// Document deleted
    DocumentDeleted,
    /// Experiment created
    ExperimentCreated,
    /// Experiment started
    ExperimentStarted,
    /// Experiment completed
    ExperimentCompleted,
    /// Task created
    TaskCreated,
    /// Task assigned
    TaskAssigned,
    /// Task completed
    TaskCompleted,
    /// Message posted
    MessagePosted,
    /// File uploaded
    FileUploaded,
    /// Merge request created
    MergeRequestCreated,
    /// Settings changed
    SettingsChanged,
}

/// Collaboration manager
#[derive(Debug)]
pub struct CollaborationManager {
    /// Active workspaces
    workspaces: HashMap<String, CollaborativeWorkspace>,
    /// Storage directory
    storage_dir: PathBuf,
    /// Manager settings
    settings: CollaborationManagerSettings,
}

/// Manager settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationManagerSettings {
    /// Maximum workspaces per user
    pub max_workspaces_per_user: u32,
    /// Default workspace settings
    pub default_workspace_settings: WorkspaceSettings,
    /// Backup settings
    pub backup_settings: BackupSettings,
}

/// Backup settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupSettings {
    /// Backup enabled
    pub enabled: bool,
    /// Backup frequency (hours)
    pub frequency_hours: u32,
    /// Backup retention (days)
    pub retention_days: u32,
    /// Backup location
    pub backup_location: PathBuf,
}

impl CollaborativeWorkspace {
    /// Create a new collaborative workspace
    pub fn new(name: &str, owner: UserInfo) -> Self {
        let now = Utc::now();
        let workspace_id = uuid::Uuid::new_v4().to_string();
        let owner_id = uuid::Uuid::new_v4().to_string();

        let owner_member = ProjectMember {
            id: owner_id.clone(),
            user: owner,
            role: MemberRole::Owner,
            permissions: vec![
                Permission::Read,
                Permission::Write,
                Permission::Delete,
                Permission::ManageMembers,
                Permission::ManagePermissions,
                Permission::ManageSettings,
                Permission::CreateExperiments,
                Permission::RunExperiments,
                Permission::PublishResults,
                Permission::AccessSensitiveData,
            ],
            joined_at: now,
            last_active: now,
            status: MemberStatus::Active,
            contributions: ContributionStats::default(),
        };

        Self {
            id: workspace_id,
            name: name.to_string(),
            members: vec![owner_member],
            documents: Vec::new(),
            channels: Vec::new(),
            tasks: Vec::new(),
            version_control: VersionControl::default(),
            settings: WorkspaceSettings::default(),
            access_control: AccessControl::default(),
            activity_log: Vec::new(),
            created_at: now,
            modified_at: now,
        }
    }

    /// Add a member to the workspace
    pub fn add_member(&mut self, user: UserInfo, role: MemberRole) -> Result<()> {
        let member_id = uuid::Uuid::new_v4().to_string();
        let permissions = self.get_default_permissions_for_role(&role);

        let member = ProjectMember {
            id: member_id.clone(),
            user,
            role,
            permissions,
            joined_at: Utc::now(),
            last_active: Utc::now(),
            status: MemberStatus::Active,
            contributions: ContributionStats::default(),
        };

        self.members.push(member);
        self.log_activity(
            &member_id,
            ActivityType::UserJoined,
            "User joined the workspace".to_string(),
            vec![],
        );

        Ok(())
    }

    /// Create a shared document
    pub fn create_document(
        &mut self,
        name: &str,
        document_type: DocumentType,
        ownerid: &str,
    ) -> Result<String> {
        if !self.has_permission(ownerid, &Permission::Write) {
            return Err(OptimError::InvalidConfig(
                "Insufficient permissions to create document".to_string(),
            ));
        }

        let document_id = uuid::Uuid::new_v4().to_string();
        let document = SharedDocument {
            id: document_id.clone(),
            name: name.to_string(),
            document_type,
            content: String::new(),
            owner_id: ownerid.to_string(),
            collaborators: Vec::new(),
            version: 1,
            version_history: Vec::new(),
            access_permissions: DocumentPermissions {
                public: false,
                read_access: self.members.iter().map(|m| m.id.clone()).collect(),
                write_access: vec![ownerid.to_string()],
                admin_access: vec![ownerid.to_string()],
            },
            metadata: DocumentMetadata {
                tags: Vec::new(),
                word_count: 0,
                character_count: 0,
                collaborator_count: 1,
                version_count: 1,
                last_editor_id: ownerid.to_string(),
            },
            created_at: Utc::now(),
            modified_at: Utc::now(),
        };

        self.documents.push(document);
        self.log_activity(
            ownerid,
            ActivityType::DocumentCreated,
            format!("Created document: {name}"),
            vec![document_id.clone()],
        );

        Ok(document_id)
    }

    /// Create a communication channel
    pub fn create_channel(
        &mut self,
        name: &str,
        channel_type: ChannelType,
        creatorid: &str,
    ) -> Result<String> {
        let channel_id = uuid::Uuid::new_v4().to_string();
        let channel = CommunicationChannel {
            id: channel_id.clone(),
            name: name.to_string(),
            description: String::new(),
            channel_type,
            members: self.members.iter().map(|m| m.id.clone()).collect(),
            messages: Vec::new(),
            settings: ChannelSettings::default(),
            created_at: Utc::now(),
        };

        self.channels.push(channel);
        Ok(channel_id)
    }

    /// Create a task
    pub fn create_task(
        &mut self,
        title: &str,
        task_type: TaskType,
        creatorid: &str,
    ) -> Result<String> {
        let task_id = uuid::Uuid::new_v4().to_string();
        let task = Task {
            id: task_id.clone(),
            title: title.to_string(),
            description: String::new(),
            task_type,
            status: TaskStatus::NotStarted,
            priority: TaskPriority::Medium,
            assigned_to: Vec::new(),
            created_by: creatorid.to_string(),
            due_date: None,
            estimated_hours: None,
            actual_hours: None,
            dependencies: Vec::new(),
            subtasks: Vec::new(),
            comments: Vec::new(),
            attachments: Vec::new(),
            labels: Vec::new(),
            created_at: Utc::now(),
            completed_at: None,
        };

        self.tasks.push(task);
        self.log_activity(
            creatorid,
            ActivityType::TaskCreated,
            format!("Created task: {title}"),
            vec![task_id.clone()],
        );

        Ok(task_id)
    }

    /// Check if a user has a specific permission
    pub fn has_permission(&self, userid: &str, permission: &Permission) -> bool {
        if let Some(member) = self.members.iter().find(|m| m.id == userid) {
            member.permissions.contains(permission)
        } else {
            false
        }
    }

    /// Log an activity
    pub fn log_activity(
        &mut self,
        user_id: &str,
        activitytype: ActivityType,
        description: String,
        resources: Vec<String>,
    ) {
        let activity = Activity {
            id: uuid::Uuid::new_v4().to_string(),
            user_id: user_id.to_string(),
            activity_type: activitytype,
            description,
            resources,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
        };

        self.activity_log.push(activity);
        self.modified_at = Utc::now();
    }

    fn get_default_permissions_for_role(&self, role: &MemberRole) -> Vec<Permission> {
        match role {
            MemberRole::Owner | MemberRole::Admin => vec![
                Permission::Read,
                Permission::Write,
                Permission::Delete,
                Permission::ManageMembers,
                Permission::ManagePermissions,
                Permission::ManageSettings,
                Permission::CreateExperiments,
                Permission::RunExperiments,
                Permission::PublishResults,
                Permission::AccessSensitiveData,
            ],
            MemberRole::PrincipalInvestigator | MemberRole::SeniorResearcher => vec![
                Permission::Read,
                Permission::Write,
                Permission::CreateExperiments,
                Permission::RunExperiments,
                Permission::PublishResults,
                Permission::AccessSensitiveData,
            ],
            MemberRole::Researcher | MemberRole::PhDStudent => vec![
                Permission::Read,
                Permission::Write,
                Permission::CreateExperiments,
                Permission::RunExperiments,
            ],
            MemberRole::MastersStudent | MemberRole::ResearchAssistant => vec![
                Permission::Read,
                Permission::Write,
                Permission::CreateExperiments,
            ],
            MemberRole::Collaborator => vec![Permission::Read, Permission::Write],
            MemberRole::Guest | MemberRole::Observer => vec![Permission::Read],
        }
    }

    /// Generate workspace statistics
    pub fn generate_statistics(&self) -> WorkspaceStatistics {
        let active_members = self
            .members
            .iter()
            .filter(|m| m.status == MemberStatus::Active)
            .count();

        let total_documents = self.documents.len();
        let total_tasks = self.tasks.len();
        let completed_tasks = self
            .tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Completed)
            .count();

        let total_messages = self.channels.iter().map(|c| c.messages.len()).sum();

        let activity_last_30_days = self
            .activity_log
            .iter()
            .filter(|a| {
                let thirty_days_ago = Utc::now() - chrono::Duration::days(30);
                a.timestamp > thirty_days_ago
            })
            .count();

        WorkspaceStatistics {
            total_members: self.members.len(),
            active_members,
            total_documents,
            total_tasks,
            completed_tasks,
            task_completion_rate: if total_tasks > 0 {
                completed_tasks as f64 / total_tasks as f64
            } else {
                0.0
            },
            total_messages,
            total_channels: self.channels.len(),
            activity_last_30_days,
            creation_date: self.created_at,
            last_activity: self.modified_at,
        }
    }
}

/// Workspace statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceStatistics {
    /// Total number of members
    pub total_members: usize,
    /// Number of active members
    pub active_members: usize,
    /// Total number of documents
    pub total_documents: usize,
    /// Total number of tasks
    pub total_tasks: usize,
    /// Number of completed tasks
    pub completed_tasks: usize,
    /// Task completion rate (0.0 to 1.0)
    pub task_completion_rate: f64,
    /// Total number of messages
    pub total_messages: usize,
    /// Total number of channels
    pub total_channels: usize,
    /// Activity count in last 30 days
    pub activity_last_30_days: usize,
    /// Workspace creation date
    pub creation_date: DateTime<Utc>,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
}

impl Default for ContributionStats {
    fn default() -> Self {
        Self {
            experiments_created: 0,
            experiments_run: 0,
            lines_of_code: 0,
            documents_authored: 0,
            comments_posted: 0,
            reviews_conducted: 0,
            contribution_score: 0.0,
        }
    }
}

impl Default for VersionControl {
    fn default() -> Self {
        Self {
            repository_url: None,
            current_branch: "main".to_string(),
            branches: vec!["main".to_string()],
            commits: Vec::new(),
            merge_requests: Vec::new(),
            settings: VersionControlSettings::default(),
        }
    }
}

impl Default for VersionControlSettings {
    fn default() -> Self {
        Self {
            auto_commit: false,
            auto_commit_frequency: 60, // 1 hour
            require_review: true,
            protected_branches: vec!["main".to_string(), "master".to_string()],
            automatic_backups: true,
        }
    }
}

impl Default for WorkspaceSettings {
    fn default() -> Self {
        Self {
            timezone: "UTC".to_string(),
            default_language: "en".to_string(),
            collaboration: CollaborationSettings::default(),
            notifications: NotificationSettings::default(),
            integrations: IntegrationSettings::default(),
        }
    }
}

impl Default for CollaborationSettings {
    fn default() -> Self {
        Self {
            real_time_editing: true,
            auto_save_frequency: 30, // 30 seconds
            conflict_resolution: ConflictResolution::Manual,
            max_simultaneous_editors: 10,
        }
    }
}

impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            email: true,
            in_app: true,
            desktop: false,
            mobile_push: false,
            frequency: NotificationFrequency::Daily,
        }
    }
}

impl Default for AccessControl {
    fn default() -> Self {
        Self {
            acls: Vec::new(),
            default_permissions: vec![Permission::Read],
            guest_access: false,
            public_visibility: false,
            invitation_settings: InvitationSettings::default(),
        }
    }
}

impl Default for InvitationSettings {
    fn default() -> Self {
        Self {
            require_approval: true,
            allow_external: false,
            expiration_days: 7,
            max_invitations_per_user: 10,
        }
    }
}

impl Default for ChannelSettings {
    fn default() -> Self {
        Self {
            notifications: true,
            auto_archive: false,
            archive_after_days: 365,
            allow_external_invites: false,
            moderation: ModerationSettings::default(),
        }
    }
}

impl Default for ModerationSettings {
    fn default() -> Self {
        Self {
            require_approval: false,
            auto_delete_inappropriate: false,
            spam_filtering: true,
            moderators: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_creation() {
        let owner = UserInfo {
            name: "Dr. Test".to_string(),
            email: "test@example.com".to_string(),
            institution: "Test University".to_string(),
            avatar_url: None,
            timezone: "UTC".to_string(),
            language: "en".to_string(),
            research_interests: vec!["machine learning".to_string()],
        };

        let workspace = CollaborativeWorkspace::new("Test Workspace", owner);

        assert_eq!(workspace.name, "Test Workspace");
        assert_eq!(workspace.members.len(), 1);
        assert_eq!(workspace.members[0].role, MemberRole::Owner);
        assert!(workspace.members[0]
            .permissions
            .contains(&Permission::ManageMembers));
    }

    #[test]
    fn test_document_creation() {
        let owner = UserInfo {
            name: "Dr. Test".to_string(),
            email: "test@example.com".to_string(),
            institution: "Test University".to_string(),
            avatar_url: None,
            timezone: "UTC".to_string(),
            language: "en".to_string(),
            research_interests: vec![],
        };

        let mut workspace = CollaborativeWorkspace::new("Test Workspace", owner);
        let owner_id = workspace.members[0].id.clone();

        let doc_id = workspace
            .create_document("Test Document", DocumentType::Manuscript, &owner_id)
            .unwrap();

        assert_eq!(workspace.documents.len(), 1);
        assert_eq!(workspace.documents[0].name, "Test Document");
        assert_eq!(
            workspace.documents[0].document_type,
            DocumentType::Manuscript
        );
        assert_eq!(workspace.activity_log.len(), 1);
        assert_eq!(
            workspace.activity_log[0].activity_type,
            ActivityType::DocumentCreated
        );
    }

    #[test]
    fn test_permission_checking() {
        let owner = UserInfo {
            name: "Dr. Test".to_string(),
            email: "test@example.com".to_string(),
            institution: "Test University".to_string(),
            avatar_url: None,
            timezone: "UTC".to_string(),
            language: "en".to_string(),
            research_interests: vec![],
        };

        let workspace = CollaborativeWorkspace::new("Test Workspace", owner);
        let owner_id = &workspace.members[0].id;

        assert!(workspace.has_permission(owner_id, &Permission::Read));
        assert!(workspace.has_permission(owner_id, &Permission::Write));
        assert!(workspace.has_permission(owner_id, &Permission::ManageMembers));

        // Test non-existent user
        assert!(!workspace.has_permission("non-existent", &Permission::Read));
    }
}
