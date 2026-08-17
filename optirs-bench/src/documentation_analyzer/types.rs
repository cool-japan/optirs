//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::path::PathBuf;

/// Information about failed examples
#[derive(Debug, Clone)]
pub struct FailedExample {
    /// Example name or identifier
    pub name: String,
    /// Source file path
    pub file_path: PathBuf,
    /// Line number where example starts
    pub line_number: usize,
    /// Compilation error message
    pub error_message: String,
    /// Example source code
    pub source_code: String,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}
/// Coverage summary
#[derive(Debug)]
pub struct CoverageSummary {
    pub percentage: f64,
    pub total_items: usize,
    pub documented_items: usize,
    pub critical_missing: Vec<String>,
}
/// Actionable recommendation
#[derive(Debug)]
pub struct ActionableRecommendation {
    pub priority: Priority,
    pub category: String,
    pub title: String,
    pub description: String,
    pub action_steps: Vec<String>,
    pub estimated_effort_hours: f64,
    pub expected_impact: f64,
}
/// Example verification results
#[derive(Debug, Default)]
pub struct ExampleVerificationResults {
    /// Total examples found
    pub total_examples: usize,
    /// Successfully compiled examples
    pub compiled_examples: usize,
    /// Failed examples with error details
    pub failed_examples: Vec<FailedExample>,
    /// Example coverage by module
    pub example_coverage: HashMap<String, ExampleCoverage>,
    /// Example quality metrics
    pub quality_metrics: ExampleQualityMetrics,
}
/// Style consistency analysis
#[derive(Debug)]
pub struct StyleAnalysis {
    /// Overall style consistency score
    pub consistency_score: f64,
    /// Style violations by category
    pub violations: HashMap<StyleCategory, Vec<StyleViolation>>,
    /// Recommended style improvements
    pub recommendations: Vec<StyleRecommendation>,
    /// Documentation format analysis
    pub format_analysis: FormatAnalysis,
}
/// Accessibility compliance analysis
#[derive(Debug, Clone)]
pub struct AccessibilityAnalysis {
    /// Alt text coverage for images
    pub alttext_coverage: f64,
    /// Color contrast compliance
    pub color_contrast_compliance: f64,
    /// Screen reader compatibility
    pub screen_reader_compatibility: f64,
    /// Keyboard navigation support
    pub keyboard_navigation_support: f64,
    /// Overall accessibility score
    pub overall_accessibility_score: f64,
}
/// Example complexity levels
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExampleComplexity {
    /// Simple usage example
    Basic,
    /// Intermediate usage with multiple features
    Intermediate,
    /// Advanced usage with complex scenarios
    Advanced,
    /// Integration examples
    Integration,
}
/// Quality assessment
#[derive(Debug)]
pub struct QualityAssessment {
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub improvement_priorities: Vec<String>,
}
/// Documentation debt categories
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DebtCategory {
    /// Missing documentation
    MissingDocumentation,
    /// Outdated documentation
    OutdatedDocumentation,
    /// Poor quality documentation
    PoorQuality,
    /// Missing examples
    MissingExamples,
    /// Broken references
    BrokenReferences,
}
/// Example quality metrics
#[derive(Debug, Clone)]
pub struct ExampleQualityMetrics {
    /// Average example length (lines)
    pub average_length: f64,
    /// Examples with proper error handling
    pub error_handling_coverage: f64,
    /// Examples with comprehensive comments
    pub comment_coverage: f64,
    /// Examples demonstrating best practices
    pub best_practices_score: f64,
}
/// Types of API changes
#[derive(Debug, Clone)]
pub enum ChangeType {
    /// Signature change
    SignatureChange,
    /// Behavior change
    BehaviorChange,
    /// Performance change
    PerformanceChange,
    /// Error handling change
    ErrorHandlingChange,
}
/// Information about style violations
#[derive(Debug, Clone)]
pub struct StyleViolation {
    /// File path where violation occurs
    pub file_path: PathBuf,
    /// Line number
    pub line_number: usize,
    /// Violation description
    pub description: String,
    /// Current content
    pub current_content: String,
    /// Suggested improvement
    pub suggested_fix: String,
    /// Severity level
    pub severity: ViolationSeverity,
}
/// API evolution tracking
#[derive(Debug, Clone, Default)]
pub struct ApiEvolutionAnalysis {
    /// New APIs since last analysis
    pub new_apis: Vec<String>,
    /// Deprecated APIs
    pub deprecated_apis: Vec<String>,
    /// Changed API signatures
    pub changed_apis: Vec<ApiChange>,
    /// Breaking changes
    pub breaking_changes: Vec<BreakingChange>,
}
/// Priority levels
#[derive(Debug, Clone)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}
/// User satisfaction metrics
#[derive(Debug, Clone)]
pub struct UserSatisfactionMetrics {
    /// Clarity score (user feedback)
    pub clarity_score: f64,
    /// Completeness score (user feedback)
    pub completeness_score: f64,
    /// Helpfulness score (user feedback)
    pub helpfulness_score: f64,
    /// Overall satisfaction score
    pub overall_satisfaction: f64,
}
/// Types of link errors
#[derive(Debug, Clone)]
pub enum LinkErrorType {
    /// HTTP error (404, 500, etc.)
    HttpError(u16),
    /// Network timeout
    Timeout,
    /// Invalid URL format
    InvalidFormat,
    /// Missing internal reference
    MissingReference,
    /// Circular reference
    CircularReference,
}
/// Effort estimate
#[derive(Debug)]
pub struct EffortEstimate {
    pub total_hours: f64,
    pub by_category: Vec<(String, f64)>,
    pub confidence_level: f64,
}
/// Style improvement recommendations
#[derive(Debug, Clone)]
pub struct StyleRecommendation {
    /// Recommendation category
    pub category: StyleCategory,
    /// Recommendation description
    pub description: String,
    /// Implementation steps
    pub implementation_steps: Vec<String>,
    /// Expected impact
    pub expected_impact: f64,
}
/// Comprehensive documentation report
#[derive(Debug)]
pub struct DocumentationReport {
    pub analysis_timestamp: std::time::SystemTime,
    pub overall_score: f64,
    pub coverage_summary: CoverageSummary,
    pub quality_assessment: QualityAssessment,
    pub actionable_recommendations: Vec<ActionableRecommendation>,
    pub estimated_effort: EffortEstimate,
}
/// Documentation debt assessment
#[derive(Debug, Clone)]
pub struct DocumentationDebt {
    /// Total debt score
    pub total_debt_score: f64,
    /// Debt by category
    pub debt_by_category: HashMap<DebtCategory, f64>,
    /// High-priority debt items
    pub high_priority_items: Vec<DebtItem>,
    /// Estimated effort to resolve debt
    pub estimated_effort_hours: f64,
}
/// Style violation categories
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StyleCategory {
    /// Inconsistent heading styles
    HeadingStyle,
    /// Inconsistent code formatting
    CodeFormatting,
    /// Missing or inconsistent parameter documentation
    ParameterStyle,
    /// Inconsistent error documentation
    ErrorStyle,
    /// Inconsistent example formatting
    ExampleStyle,
    /// Language and tone inconsistencies
    LanguageStyle,
}
/// Visibility levels
#[derive(Debug, Clone)]
pub enum VisibilityLevel {
    Public,
    PublicCrate,
    PublicSuper,
    Private,
}
/// API completeness analysis
#[derive(Debug, Default)]
pub struct ApiCompletenessAnalysis {
    /// Missing documentation sections
    pub missing_sections: HashMap<String, Vec<MissingSection>>,
    /// API evolution tracking
    pub evolution_tracking: ApiEvolutionAnalysis,
    /// Documentation debt assessment
    pub documentation_debt: DocumentationDebt,
    /// Accessibility compliance
    pub accessibility_compliance: AccessibilityAnalysis,
}
/// API change information
#[derive(Debug, Clone)]
pub struct ApiChange {
    /// API name
    pub api_name: String,
    /// Change type
    pub change_type: ChangeType,
    /// Description of change
    pub description: String,
    /// Documentation update status
    pub documentation_updated: bool,
}
/// Categories of API items
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ItemCategory {
    /// Public functions
    Function,
    /// Public structs
    Struct,
    /// Public enums
    Enum,
    /// Public traits
    Trait,
    /// Public modules
    Module,
    /// Public constants
    Constant,
    /// Public macros
    Macro,
}
/// Documentation format analysis
#[derive(Debug, Clone)]
pub struct FormatAnalysis {
    /// Markdown compliance score
    pub markdown_compliance: f64,
    /// Rustdoc compliance score
    pub rustdoc_compliance: f64,
    /// Cross-reference completeness
    pub cross_reference_completeness: f64,
    /// Table of contents quality
    pub toc_quality: f64,
}
/// Breaking change information
#[derive(Debug, Clone)]
pub struct BreakingChange {
    /// API name
    pub api_name: String,
    /// Change description
    pub description: String,
    /// Migration guide availability
    pub migration_guide_available: bool,
    /// Suggested migration path
    pub migration_path: String,
}
/// Link status
#[derive(Debug, Clone)]
pub enum LinkStatus {
    Valid,
    Broken(String),
    Redirected(String),
    Timeout,
}
/// Documentation coverage analysis
#[derive(Debug)]
pub struct CoverageAnalysis {
    /// Total public items
    pub total_public_items: usize,
    /// Documented public items
    pub documented_items: usize,
    /// Coverage percentage
    pub coverage_percentage: f64,
    /// Undocumented items by category
    pub undocumented_by_category: HashMap<ItemCategory, Vec<UndocumentedItem>>,
    /// Documentation quality scores by module
    pub quality_by_module: HashMap<String, f64>,
}
/// Example coverage metrics for a module
#[derive(Debug, Clone)]
pub struct ExampleCoverage {
    /// Public functions with examples
    pub functions_with_examples: usize,
    /// Total public functions
    pub total_functions: usize,
    /// Example coverage percentage
    pub coverage_percentage: f64,
    /// Example complexity levels
    pub complexity_distribution: HashMap<ExampleComplexity, usize>,
}
/// Complete analysis results
#[derive(Debug)]
pub struct AnalysisResults {
    /// Documentation coverage analysis
    pub coverage: CoverageAnalysis,
    /// Example verification results
    pub example_verification: ExampleVerificationResults,
    /// Link checking results
    pub link_checking: LinkCheckingResults,
    /// Style consistency analysis
    pub style_analysis: StyleAnalysis,
    /// API completeness assessment
    pub api_completeness: ApiCompletenessAnalysis,
    /// Quality score (0.0 to 1.0)
    pub overall_quality_score: f64,
}
/// Information about broken links
#[derive(Debug, Clone)]
pub struct BrokenLink {
    /// Link URL or path
    pub url: String,
    /// Source file where link was found
    pub source_file: PathBuf,
    /// Line number
    pub line_number: usize,
    /// Error type
    pub error_type: LinkErrorType,
    /// Error message
    pub error_message: String,
    /// Suggested replacement
    pub suggested_replacement: Option<String>,
}
/// Link checking results
#[derive(Debug)]
pub struct LinkCheckingResults {
    /// Total links checked
    pub total_links: usize,
    /// Valid links
    pub valid_links: usize,
    /// Broken links with details
    pub broken_links: Vec<BrokenLink>,
    /// External link status
    pub external_link_status: HashMap<String, LinkStatus>,
    /// Internal link consistency
    pub internal_link_consistency: f64,
}
/// Violation severity levels
#[derive(Debug, Clone)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}
/// Information about undocumented items
#[derive(Debug, Clone)]
pub struct UndocumentedItem {
    /// Item name
    pub name: String,
    /// File path
    pub file_path: PathBuf,
    /// Line number
    pub line_number: usize,
    /// Item category
    pub category: ItemCategory,
    /// Visibility level
    pub visibility: VisibilityLevel,
    /// Suggested documentation template
    pub suggested_template: String,
}
/// Information about missing documentation sections
#[derive(Debug, Clone)]
pub struct MissingSection {
    /// Section name
    pub section_name: String,
    /// Item name where section is missing
    pub item_name: String,
    /// File path
    pub file_path: PathBuf,
    /// Priority for adding this section
    pub priority: Priority,
    /// Template for the missing section
    pub suggested_template: String,
}
/// Documentation debt item
#[derive(Debug, Clone)]
pub struct DebtItem {
    /// Debt category
    pub category: DebtCategory,
    /// Description
    pub description: String,
    /// File path
    pub file_path: PathBuf,
    /// Priority
    pub priority: Priority,
    /// Estimated effort (hours)
    pub estimated_effort: f64,
}
/// Documentation metrics
#[derive(Debug)]
pub struct DocumentationMetrics {
    /// Total lines of documentation
    pub total_doc_lines: usize,
    /// Lines of code vs documentation ratio
    pub code_to_doc_ratio: f64,
    /// Average documentation quality score
    pub average_quality_score: f64,
    /// Documentation maintenance burden
    pub maintenance_burden: f64,
    /// User satisfaction metrics
    pub user_satisfaction: UserSatisfactionMetrics,
}
/// Configuration for documentation analysis
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    /// Source directories to analyze
    pub source_directories: Vec<PathBuf>,
    /// Documentation output directory
    pub docs_output_dir: PathBuf,
    /// Minimum documentation coverage required
    pub min_coverage_threshold: f64,
    /// Check for example code compilation
    pub verify_examples: bool,
    /// Check for broken links
    pub check_links: bool,
    /// Analyze documentation style consistency
    pub check_style_consistency: bool,
    /// Required documentation sections
    pub required_sections: Vec<String>,
    /// Documentation language preferences
    pub language_preferences: Vec<String>,
}
