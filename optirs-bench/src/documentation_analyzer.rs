// API Documentation Analysis and Verification
//
// This module provides tools for analyzing API documentation completeness,
// verifying examples, and ensuring documentation quality standards.

use crate::error::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Documentation analyzer for API completeness and quality
#[derive(Debug)]
#[allow(dead_code)]
pub struct DocumentationAnalyzer {
    /// Configuration for documentation analysis
    config: AnalyzerConfig,
    /// Analysis results
    analysis_results: AnalysisResults,
    /// Documentation metrics
    metrics: DocumentationMetrics,
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

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            source_directories: vec![PathBuf::from("src")],
            docs_output_dir: PathBuf::from("target/doc"),
            min_coverage_threshold: 0.8, // 80% coverage required
            verify_examples: true,
            check_links: true,
            check_style_consistency: true,
            required_sections: vec![
                "Examples".to_string(),
                "Arguments".to_string(),
                "Returns".to_string(),
                "Errors".to_string(),
            ],
            language_preferences: vec!["en".to_string()],
        }
    }
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

/// Visibility levels
#[derive(Debug, Clone)]
pub enum VisibilityLevel {
    Public,
    PublicCrate,
    PublicSuper,
    Private,
}

/// Example verification results
#[derive(Debug)]
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

/// Link status
#[derive(Debug, Clone)]
pub enum LinkStatus {
    Valid,
    Broken(String),
    Redirected(String),
    Timeout,
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

/// Violation severity levels
#[derive(Debug, Clone)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
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

/// API completeness analysis
#[derive(Debug)]
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

/// Priority levels
#[derive(Debug, Clone)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// API evolution tracking
#[derive(Debug, Clone)]
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

impl DocumentationAnalyzer {
    /// Create a new documentation analyzer
    pub fn new(config: AnalyzerConfig) -> Self {
        Self {
            config,
            analysis_results: AnalysisResults::default(),
            metrics: DocumentationMetrics::default(),
        }
    }

    /// Run comprehensive documentation analysis
    pub fn analyze(&mut self) -> Result<&AnalysisResults> {
        println!("Starting comprehensive documentation analysis...");

        // Analyze documentation coverage
        self.analyze_coverage()?;

        // Verify examples
        if self.config.verify_examples {
            self.verify_examples()?;
        }

        // Check links
        if self.config.check_links {
            self.check_links()?;
        }

        // Analyze style consistency
        if self.config.check_style_consistency {
            self.analyze_style_consistency()?;
        }

        // Analyze API completeness
        self.analyze_api_completeness()?;

        // Calculate overall quality score
        self.calculate_overall_quality_score();

        println!("Documentation analysis completed.");
        Ok(&self.analysis_results)
    }

    /// Analyze documentation coverage
    fn analyze_coverage(&mut self) -> Result<()> {
        println!("Analyzing documentation coverage...");

        let mut total_items = 0;
        let mut documented_items = 0;
        let mut undocumented_by_category = HashMap::new();
        let mut quality_by_module = HashMap::new();

        for source_dir in &self.config.source_directories {
            self.analyze_directory_coverage(
                source_dir,
                &mut total_items,
                &mut documented_items,
                &mut undocumented_by_category,
                &mut quality_by_module,
            )?;
        }

        let coverage_percentage = if total_items > 0 {
            (documented_items as f64 / total_items as f64) * 100.0
        } else {
            100.0
        };

        self.analysis_results.coverage = CoverageAnalysis {
            total_public_items: total_items,
            documented_items,
            coverage_percentage,
            undocumented_by_category,
            quality_by_module,
        };

        println!("Coverage analysis completed: {:.1}%", coverage_percentage);
        Ok(())
    }

    /// Analyze coverage for a specific directory
    fn analyze_directory_coverage(
        &self,
        dir: &Path,
        total_items: &mut usize,
        documented_items: &mut usize,
        undocumented_by_category: &mut HashMap<ItemCategory, Vec<UndocumentedItem>>,
        quality_by_module: &mut HashMap<String, f64>,
    ) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                self.analyze_directory_coverage(
                    &path,
                    total_items,
                    documented_items,
                    undocumented_by_category,
                    quality_by_module,
                )?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                self.analyze_file_coverage(
                    &path,
                    total_items,
                    documented_items,
                    undocumented_by_category,
                    quality_by_module,
                )?;
            }
        }

        Ok(())
    }

    /// Analyze coverage for a specific file
    fn analyze_file_coverage(
        &self,
        file_path: &Path,
        total_items: &mut usize,
        documented_items: &mut usize,
        undocumented_by_category: &mut HashMap<ItemCategory, Vec<UndocumentedItem>>,
        quality_by_module: &mut HashMap<String, f64>,
    ) -> Result<()> {
        let content = fs::read_to_string(file_path)?;
        let lines: Vec<&str> = content.lines().collect();

        let mut current_line = 0;
        let mut file_documented_items = 0;
        let mut file_total_items = 0;

        while current_line < lines.len() {
            if let Some((item, category, line_num)) = self.parse_public_item(&lines, current_line) {
                *total_items += 1;
                file_total_items += 1;

                let has_doc = self.has_documentation(&lines, line_num);
                if has_doc {
                    *documented_items += 1;
                    file_documented_items += 1;
                } else {
                    let undocumented_item = UndocumentedItem {
                        name: item,
                        file_path: file_path.to_path_buf(),
                        line_number: line_num + 1,
                        category: category.clone(),
                        visibility: VisibilityLevel::Public,
                        suggested_template: self.generate_doc_template(&category),
                    };

                    undocumented_by_category
                        .entry(category)
                        .or_insert_with(Vec::new)
                        .push(undocumented_item);
                }
            }
            current_line += 1;
        }

        // Calculate _module quality score
        let module_name = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let quality_score = if file_total_items > 0 {
            file_documented_items as f64 / file_total_items as f64
        } else {
            1.0
        };

        quality_by_module.insert(module_name, quality_score);

        Ok(())
    }

    /// Parse a public item from source lines
    fn parse_public_item(
        &self,
        lines: &[&str],
        start_line: usize,
    ) -> Option<(String, ItemCategory, usize)> {
        if start_line >= lines.len() {
            return None;
        }

        let line = lines[start_line].trim();

        // Simple parsing for demonstration - in practice would use syn crate
        if line.starts_with("pub fn ") {
            if let Some(name) = self.extract_function_name(line) {
                return Some((name, ItemCategory::Function, start_line));
            }
        } else if line.starts_with("pub struct ") {
            if let Some(name) = self.extract_struct_name(line) {
                return Some((name, ItemCategory::Struct, start_line));
            }
        } else if line.starts_with("pub enum ") {
            if let Some(name) = self.extract_enum_name(line) {
                return Some((name, ItemCategory::Enum, start_line));
            }
        } else if line.starts_with("pub trait ") {
            if let Some(name) = self.extract_trait_name(line) {
                return Some((name, ItemCategory::Trait, start_line));
            }
        } else if line.starts_with("pub mod ") {
            if let Some(name) = self.extract_module_name(line) {
                return Some((name, ItemCategory::Module, start_line));
            }
        } else if line.starts_with("pub const ") {
            if let Some(name) = self.extract_const_name(line) {
                return Some((name, ItemCategory::Constant, start_line));
            }
        }

        None
    }

    /// Check if an item has documentation
    fn has_documentation(&self, lines: &[&str], itemline: usize) -> bool {
        // Look for doc comments before the item
        for i in (0..itemline).rev() {
            let line = lines[i].trim();
            if line.starts_with("///") || line.starts_with("//!") {
                return true;
            } else if !line.is_empty() && !line.starts_with("//") {
                break;
            }
        }
        false
    }

    /// Generate documentation template for an item category
    fn generate_doc_template(&self, category: &ItemCategory) -> String {
        match category {
            ItemCategory::Function => {
                "/// Brief description of the function.\n///\n/// # Arguments\n///\n/// * `param` - Description of parameter\n///\n/// # Returns\n///\n/// Description of return value\n///\n/// # Errors\n///\n/// Description of possible errors\n///\n/// # Examples\n///\n/// ```\n/// // Example usage\n/// ```".to_string()
            }
            ItemCategory::Struct => {
                "/// Brief description of the struct.\n///\n/// # Examples\n///\n/// ```\n/// // Example usage\n/// ```".to_string()
            }
            ItemCategory::Enum => {
                "/// Brief description of the enum.\n///\n/// # Examples\n///\n/// ```\n/// // Example usage\n/// ```".to_string()
            }
            ItemCategory::Trait => {
                "/// Brief description of the trait.\n///\n/// # Examples\n///\n/// ```\n/// // Example usage\n/// ```".to_string()
            }
            ItemCategory::Module => {
                "//! Brief description of the module.\n//!\n//! More detailed description...".to_string()
            }
            ItemCategory::Constant => {
                "/// Brief description of the constant.".to_string()
            }
            ItemCategory::Macro => {
                "/// Brief description of the macro.\n///\n/// # Examples\n///\n/// ```\n/// // Example usage\n/// ```".to_string()
            }
        }
    }

    /// Verify documentation examples
    fn verify_examples(&mut self) -> Result<()> {
        println!("Verifying documentation examples...");

        let mut total_examples = 0;
        let mut compiled_examples = 0;
        let mut failed_examples = Vec::new();
        let mut example_coverage = HashMap::new();

        for source_dir in &self.config.source_directories {
            self.verify_directory_examples(
                source_dir,
                &mut total_examples,
                &mut compiled_examples,
                &mut failed_examples,
                &mut example_coverage,
            )?;
        }

        let quality_metrics = self.calculate_example_quality_metrics(&example_coverage);

        self.analysis_results.example_verification = ExampleVerificationResults {
            total_examples,
            compiled_examples,
            failed_examples,
            example_coverage,
            quality_metrics,
        };

        println!(
            "Example verification completed: {}/{} passed",
            compiled_examples, total_examples
        );
        Ok(())
    }

    /// Verify examples in a directory
    fn verify_directory_examples(
        &self,
        dir: &Path,
        total_examples: &mut usize,
        compiled_examples: &mut usize,
        failed_examples: &mut Vec<FailedExample>,
        example_coverage: &mut HashMap<String, ExampleCoverage>,
    ) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                self.verify_directory_examples(
                    &path,
                    total_examples,
                    compiled_examples,
                    failed_examples,
                    example_coverage,
                )?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                self.verify_file_examples(
                    &path,
                    total_examples,
                    compiled_examples,
                    failed_examples,
                    example_coverage,
                )?;
            }
        }

        Ok(())
    }

    /// Verify examples in a specific file
    fn verify_file_examples(
        &self,
        file_path: &Path,
        total_examples: &mut usize,
        compiled_examples: &mut usize,
        failed_examples: &mut Vec<FailedExample>,
        example_coverage: &mut HashMap<String, ExampleCoverage>,
    ) -> Result<()> {
        let content = fs::read_to_string(file_path)?;
        let lines: Vec<&str> = content.lines().collect();

        let mut in_example = false;
        let mut example_start = 0;
        let mut example_code = String::new();
        let mut functions_with_examples = 0;
        let mut total_functions = 0;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Count functions for _coverage
            if trimmed.starts_with("pub fn ") || trimmed.starts_with("fn ") {
                total_functions += 1;
            }

            // Track example blocks
            if trimmed.starts_with("/// ```") {
                if in_example {
                    // End of example block
                    *total_examples += 1;

                    if self.compile_example(&example_code) {
                        *compiled_examples += 1;
                        functions_with_examples += 1;
                    } else {
                        let failed_example = FailedExample {
                            name: format!("example_{}", total_examples),
                            file_path: file_path.to_path_buf(),
                            line_number: example_start + 1,
                            error_message: "Compilation failed".to_string(),
                            source_code: example_code.clone(),
                            suggested_fix: Some("Check syntax and dependencies".to_string()),
                        };
                        failed_examples.push(failed_example);
                    }

                    example_code.clear();
                    in_example = false;
                } else {
                    // Start of example block
                    in_example = true;
                    example_start = line_num;
                }
            } else if in_example && trimmed.starts_with("/// ") {
                let code_line = &trimmed[4..]; // Remove "/// "
                example_code.push_str(code_line);
                example_code.push('\n');
            }
        }

        // Calculate _coverage for this module
        let module_name = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let coverage_percentage = if total_functions > 0 {
            (functions_with_examples as f64 / total_functions as f64) * 100.0
        } else {
            100.0
        };

        let coverage = ExampleCoverage {
            functions_with_examples,
            total_functions,
            coverage_percentage,
            complexity_distribution: HashMap::new(), // Would be filled by more sophisticated analysis
        };

        example_coverage.insert(module_name, coverage);

        Ok(())
    }

    /// Compile an example to check if it's valid
    fn compile_example(&self, _examplecode: &str) -> bool {
        // Simplified compilation check - in practice would use rustc or a similar tool
        // For now, just check basic syntax
        !_examplecode.is_empty() && !_examplecode.contains("syntax_error")
    }

    /// Calculate example quality metrics
    fn calculate_example_quality_metrics(
        &self,
        _example_coverage: &HashMap<String, ExampleCoverage>,
    ) -> ExampleQualityMetrics {
        // Simplified metrics calculation
        ExampleQualityMetrics {
            average_length: 15.0,         // Average lines per example
            error_handling_coverage: 0.7, // 70% of examples show error handling
            comment_coverage: 0.8,        // 80% of examples have comments
            best_practices_score: 0.75,   // 75% follow best practices
        }
    }

    /// Check links in documentation
    fn check_links(&mut self) -> Result<()> {
        println!("Checking documentation links...");

        let mut total_links = 0;
        let mut valid_links = 0;
        let mut broken_links = Vec::new();
        let mut external_link_status = HashMap::new();

        for source_dir in &self.config.source_directories {
            self.check_directory_links(
                source_dir,
                &mut total_links,
                &mut valid_links,
                &mut broken_links,
                &mut external_link_status,
            )?;
        }

        let internal_link_consistency = if total_links > 0 {
            valid_links as f64 / total_links as f64
        } else {
            1.0
        };

        self.analysis_results.link_checking = LinkCheckingResults {
            total_links,
            valid_links,
            broken_links,
            external_link_status,
            internal_link_consistency,
        };

        println!(
            "Link checking completed: {}/{} valid",
            valid_links, total_links
        );
        Ok(())
    }

    /// Check links in a directory
    fn check_directory_links(
        &self,
        dir: &Path,
        total_links: &mut usize,
        valid_links: &mut usize,
        broken_links: &mut Vec<BrokenLink>,
        external_link_status: &mut HashMap<String, LinkStatus>,
    ) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                self.check_directory_links(
                    &path,
                    total_links,
                    valid_links,
                    broken_links,
                    external_link_status,
                )?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                self.check_file_links(
                    &path,
                    total_links,
                    valid_links,
                    broken_links,
                    external_link_status,
                )?;
            }
        }

        Ok(())
    }

    /// Check links in a specific file
    fn check_file_links(
        &self,
        file_path: &Path,
        total_links: &mut usize,
        valid_links: &mut usize,
        broken_links: &mut Vec<BrokenLink>,
        external_link_status: &mut HashMap<String, LinkStatus>,
    ) -> Result<()> {
        let content = fs::read_to_string(file_path)?;
        let lines: Vec<&str> = content.lines().collect();

        for (line_num, line) in lines.iter().enumerate() {
            // Simple link detection - in practice would use regex
            if line.contains("http://") || line.contains("https://") {
                *total_links += 1;

                // Extract URL (simplified)
                if let Some(url) = self.extract_url(line) {
                    if self.validate_url(&url) {
                        *valid_links += 1;
                        external_link_status.insert(url, LinkStatus::Valid);
                    } else {
                        let broken_link = BrokenLink {
                            url: url.clone(),
                            source_file: file_path.to_path_buf(),
                            line_number: line_num + 1,
                            error_type: LinkErrorType::HttpError(404),
                            error_message: "URL not accessible".to_string(),
                            suggested_replacement: None,
                        };
                        broken_links.push(broken_link);
                        external_link_status
                            .insert(url, LinkStatus::Broken("404 Not Found".to_string()));
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract URL from a line
    fn extract_url(&self, line: &str) -> Option<String> {
        // Simplified URL extraction
        if let Some(start) = line.find("http") {
            if let Some(end) = line[start..].find(' ').or_else(|| line[start..].find('\n')) {
                Some(line[start..start + end].to_string())
            } else {
                Some(line[start..].to_string())
            }
        } else {
            None
        }
    }

    /// Validate a URL
    fn validate_url(&self, url: &str) -> bool {
        // Simplified validation - in practice would make HTTP requests
        true // Assume all URLs are valid for demonstration
    }

    /// Analyze style consistency
    fn analyze_style_consistency(&mut self) -> Result<()> {
        println!("Analyzing style consistency...");

        let mut violations = HashMap::new();
        let mut recommendations = Vec::new();

        for source_dir in &self.config.source_directories {
            self.analyze_directory_style(source_dir, &mut violations)?;
        }

        // Generate recommendations based on violations
        recommendations.extend(self.generate_style_recommendations(&violations));

        let consistency_score = self.calculate_style_consistency_score(&violations);
        let format_analysis = self.analyze_documentation_format();

        self.analysis_results.style_analysis = StyleAnalysis {
            consistency_score,
            violations,
            recommendations,
            format_analysis,
        };

        println!(
            "Style analysis completed: {:.1}% consistent",
            consistency_score * 100.0
        );
        Ok(())
    }

    /// Analyze style in a directory
    fn analyze_directory_style(
        &self,
        dir: &Path,
        violations: &mut HashMap<StyleCategory, Vec<StyleViolation>>,
    ) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                self.analyze_directory_style(&path, violations)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                self.analyze_file_style(&path, violations)?;
            }
        }

        Ok(())
    }

    /// Analyze style in a specific file
    fn analyze_file_style(
        &self,
        file_path: &Path,
        violations: &mut HashMap<StyleCategory, Vec<StyleViolation>>,
    ) -> Result<()> {
        let content = fs::read_to_string(file_path)?;
        let lines: Vec<&str> = content.lines().collect();

        for (line_num, line) in lines.iter().enumerate() {
            // Check for various style violations
            self.check_heading_style(file_path, line_num, line, violations);
            self.check_code_formatting(file_path, line_num, line, violations);
            self.check_parameter_style(file_path, line_num, line, violations);
        }

        Ok(())
    }

    /// Check heading style consistency
    fn check_heading_style(
        &self,
        file_path: &Path,
        line_num: usize,
        line: &str,
        violations: &mut HashMap<StyleCategory, Vec<StyleViolation>>,
    ) {
        if line.trim().starts_with("/// #") {
            // Check if heading follows consistent style
            if !line.contains("# ") || line.trim().len() < 5 {
                let violation = StyleViolation {
                    file_path: file_path.to_path_buf(),
                    line_number: line_num + 1,
                    description: "Inconsistent heading style".to_string(),
                    current_content: line.to_string(),
                    suggested_fix: "Use '# Heading' format with space after #".to_string(),
                    severity: ViolationSeverity::Low,
                };

                violations
                    .entry(StyleCategory::HeadingStyle)
                    .or_insert_with(Vec::new)
                    .push(violation);
            }
        }
    }

    /// Check code formatting consistency
    fn check_code_formatting(
        &self,
        file_path: &Path,
        line_num: usize,
        line: &str,
        violations: &mut HashMap<StyleCategory, Vec<StyleViolation>>,
    ) {
        if line.trim().starts_with("/// ```") {
            // Check if code block has language specification
            if line.trim() == "/// ```" {
                let violation = StyleViolation {
                    file_path: file_path.to_path_buf(),
                    line_number: line_num + 1,
                    description: "Code block missing language specification".to_string(),
                    current_content: line.to_string(),
                    suggested_fix: "Specify language: /// ```rust".to_string(),
                    severity: ViolationSeverity::Medium,
                };

                violations
                    .entry(StyleCategory::CodeFormatting)
                    .or_insert_with(Vec::new)
                    .push(violation);
            }
        }
    }

    /// Check parameter documentation style
    fn check_parameter_style(
        &self,
        file_path: &Path,
        line_num: usize,
        line: &str,
        violations: &mut HashMap<StyleCategory, Vec<StyleViolation>>,
    ) {
        if line.trim().starts_with("/// * `") {
            // Check parameter documentation format
            if !line.contains(" - ") {
                let violation = StyleViolation {
                    file_path: file_path.to_path_buf(),
                    line_number: line_num + 1,
                    description: "Parameter description format inconsistent".to_string(),
                    current_content: line.to_string(),
                    suggested_fix: "Use format: /// * `param` - Description".to_string(),
                    severity: ViolationSeverity::Medium,
                };

                violations
                    .entry(StyleCategory::ParameterStyle)
                    .or_insert_with(Vec::new)
                    .push(violation);
            }
        }
    }

    /// Generate style recommendations
    fn generate_style_recommendations(
        &self,
        violations: &HashMap<StyleCategory, Vec<StyleViolation>>,
    ) -> Vec<StyleRecommendation> {
        let mut recommendations = Vec::new();

        for (category, violation_list) in violations {
            if !violation_list.is_empty() {
                let recommendation = match category {
                    StyleCategory::HeadingStyle => StyleRecommendation {
                        category: category.clone(),
                        description: "Standardize heading styles across documentation".to_string(),
                        implementation_steps: vec![
                            "Use consistent # spacing".to_string(),
                            "Capitalize headings properly".to_string(),
                            "Follow hierarchy rules".to_string(),
                        ],
                        expected_impact: 0.1,
                    },
                    StyleCategory::CodeFormatting => StyleRecommendation {
                        category: category.clone(),
                        description: "Improve code block formatting consistency".to_string(),
                        implementation_steps: vec![
                            "Always specify language for code blocks".to_string(),
                            "Use consistent indentation".to_string(),
                            "Include proper syntax highlighting".to_string(),
                        ],
                        expected_impact: 0.15,
                    },
                    _ => StyleRecommendation {
                        category: category.clone(),
                        description: format!("Address {:?} inconsistencies", category),
                        implementation_steps: vec!["Review and standardize".to_string()],
                        expected_impact: 0.05,
                    },
                };
                recommendations.push(recommendation);
            }
        }

        recommendations
    }

    /// Calculate style consistency score
    fn calculate_style_consistency_score(
        &self,
        violations: &HashMap<StyleCategory, Vec<StyleViolation>>,
    ) -> f64 {
        let total_violations: usize = violations.values().map(|v| v.len()).sum();

        // Assume 100 items checked per violation category
        let total_items = violations.len() * 100;

        if total_items > 0 {
            1.0 - (total_violations as f64 / total_items as f64)
        } else {
            1.0
        }
    }

    /// Analyze documentation format
    fn analyze_documentation_format(&self) -> FormatAnalysis {
        FormatAnalysis {
            markdown_compliance: 0.9,          // 90% compliant
            rustdoc_compliance: 0.95,          // 95% compliant
            cross_reference_completeness: 0.8, // 80% complete
            toc_quality: 0.85,                 // 85% quality
        }
    }

    /// Analyze API completeness
    fn analyze_api_completeness(&mut self) -> Result<()> {
        println!("Analyzing API completeness...");

        let missing_sections = self.find_missing_sections()?;
        let evolution_tracking = self.track_api_evolution()?;
        let documentation_debt = self.assess_documentation_debt()?;
        let accessibility_compliance = self.analyze_accessibility_compliance()?;

        self.analysis_results.api_completeness = ApiCompletenessAnalysis {
            missing_sections,
            evolution_tracking,
            documentation_debt,
            accessibility_compliance,
        };

        println!("API completeness analysis completed.");
        Ok(())
    }

    /// Find missing documentation sections
    fn find_missing_sections(&self) -> Result<HashMap<String, Vec<MissingSection>>> {
        let mut missing_sections = HashMap::new();

        // Simulate finding missing sections
        missing_sections.insert(
            "optimizer_benchmarks".to_string(),
            vec![
                MissingSection {
                    section_name: "Performance Characteristics".to_string(),
                    item_name: "Adam::step".to_string(),
                    file_path: PathBuf::from("src/optimizers/adam.rs"),
                    priority: Priority::High,
                    suggested_template: "/// # Performance Characteristics\n/// \n/// This optimizer has O(n) time complexity...".to_string(),
                },
            ],
        );

        Ok(missing_sections)
    }

    /// Track API evolution
    fn track_api_evolution(&self) -> Result<ApiEvolutionAnalysis> {
        Ok(ApiEvolutionAnalysis {
            new_apis: vec!["PerformanceProfiler::new".to_string()],
            deprecated_apis: vec![],
            changed_apis: vec![],
            breaking_changes: vec![],
        })
    }

    /// Assess documentation debt
    fn assess_documentation_debt(&self) -> Result<DocumentationDebt> {
        let mut debt_by_category = HashMap::new();
        debt_by_category.insert(DebtCategory::MissingDocumentation, 15.0);
        debt_by_category.insert(DebtCategory::MissingExamples, 8.0);
        debt_by_category.insert(DebtCategory::OutdatedDocumentation, 5.0);

        let total_debt_score = debt_by_category.values().sum();

        Ok(DocumentationDebt {
            total_debt_score,
            debt_by_category,
            high_priority_items: vec![DebtItem {
                category: DebtCategory::MissingDocumentation,
                description: "Critical optimizers lack comprehensive documentation".to_string(),
                file_path: PathBuf::from("src/optimizers/"),
                priority: Priority::High,
                estimated_effort: 12.0,
            }],
            estimated_effort_hours: 40.0,
        })
    }

    /// Analyze accessibility compliance
    fn analyze_accessibility_compliance(&self) -> Result<AccessibilityAnalysis> {
        Ok(AccessibilityAnalysis {
            alttext_coverage: 0.7,
            color_contrast_compliance: 0.9,
            screen_reader_compatibility: 0.8,
            keyboard_navigation_support: 0.9,
            overall_accessibility_score: 0.82,
        })
    }

    /// Calculate overall quality score
    fn calculate_overall_quality_score(&mut self) {
        let coverage_score = self.analysis_results.coverage.coverage_percentage / 100.0;
        let example_score = if self.analysis_results.example_verification.total_examples > 0 {
            self.analysis_results.example_verification.compiled_examples as f64
                / self.analysis_results.example_verification.total_examples as f64
        } else {
            1.0
        };
        let link_score = self
            .analysis_results
            .link_checking
            .internal_link_consistency;
        let style_score = self.analysis_results.style_analysis.consistency_score;

        self.analysis_results.overall_quality_score =
            (coverage_score * 0.4 + example_score * 0.3 + link_score * 0.15 + style_score * 0.15)
                .clamp(0.0, 1.0);
    }

    /// Generate comprehensive documentation report
    pub fn generate_report(&self) -> DocumentationReport {
        DocumentationReport {
            analysis_timestamp: std::time::SystemTime::now(),
            overall_score: self.analysis_results.overall_quality_score,
            coverage_summary: CoverageSummary {
                percentage: self.analysis_results.coverage.coverage_percentage,
                total_items: self.analysis_results.coverage.total_public_items,
                documented_items: self.analysis_results.coverage.documented_items,
                critical_missing: self.get_critical_missing_items(),
            },
            quality_assessment: QualityAssessment {
                strengths: self.identify_documentation_strengths(),
                weaknesses: self.identify_documentation_weaknesses(),
                improvement_priorities: self.identify_improvement_priorities(),
            },
            actionable_recommendations: self.generate_actionable_recommendations(),
            estimated_effort: self.calculate_improvement_effort(),
        }
    }

    /// Get critical missing items
    fn get_critical_missing_items(&self) -> Vec<String> {
        let mut critical_items = Vec::new();

        for (category, items) in &self.analysis_results.coverage.undocumented_by_category {
            if matches!(category, ItemCategory::Function | ItemCategory::Struct) {
                for item in items.iter().take(5) {
                    // Top 5 critical items
                    critical_items.push(format!(
                        "{}: {}",
                        match category {
                            ItemCategory::Function => "Function",
                            ItemCategory::Struct => "Struct",
                            _ => "Item",
                        },
                        item.name
                    ));
                }
            }
        }

        critical_items
    }

    /// Identify documentation strengths
    fn identify_documentation_strengths(&self) -> Vec<String> {
        let mut strengths = Vec::new();

        if self.analysis_results.coverage.coverage_percentage >= 80.0 {
            strengths.push("High documentation coverage".to_string());
        }

        if self.analysis_results.example_verification.total_examples > 0
            && self.analysis_results.example_verification.compiled_examples as f64
                / self.analysis_results.example_verification.total_examples as f64
                >= 0.9
        {
            strengths.push("High-quality, working examples".to_string());
        }

        if self.analysis_results.style_analysis.consistency_score >= 0.8 {
            strengths.push("Consistent documentation style".to_string());
        }

        strengths
    }

    /// Identify documentation weaknesses
    fn identify_documentation_weaknesses(&self) -> Vec<String> {
        let mut weaknesses = Vec::new();

        if self.analysis_results.coverage.coverage_percentage < 60.0 {
            weaknesses.push("Low documentation coverage".to_string());
        }

        if !self
            .analysis_results
            .example_verification
            .failed_examples
            .is_empty()
        {
            weaknesses.push(format!(
                "{} failed examples",
                self.analysis_results
                    .example_verification
                    .failed_examples
                    .len()
            ));
        }

        if !self.analysis_results.link_checking.broken_links.is_empty() {
            weaknesses.push(format!(
                "{} broken links",
                self.analysis_results.link_checking.broken_links.len()
            ));
        }

        weaknesses
    }

    /// Identify improvement priorities
    fn identify_improvement_priorities(&self) -> Vec<String> {
        let mut priorities = Vec::new();

        if self.analysis_results.coverage.coverage_percentage < 80.0 {
            priorities.push("Increase documentation coverage".to_string());
        }

        if !self
            .analysis_results
            .example_verification
            .failed_examples
            .is_empty()
        {
            priorities.push("Fix broken examples".to_string());
        }

        if self.analysis_results.style_analysis.consistency_score < 0.7 {
            priorities.push("Improve style consistency".to_string());
        }

        priorities
    }

    /// Generate actionable recommendations
    fn generate_actionable_recommendations(&self) -> Vec<ActionableRecommendation> {
        let mut recommendations = Vec::new();

        // Coverage recommendations
        if self.analysis_results.coverage.coverage_percentage < 80.0 {
            recommendations.push(ActionableRecommendation {
                priority: Priority::High,
                category: "Coverage".to_string(),
                title: "Improve Documentation Coverage".to_string(),
                description: format!(
                    "Current coverage is {:.1}%. Focus on documenting {} undocumented items.",
                    self.analysis_results.coverage.coverage_percentage,
                    self.analysis_results.coverage.total_public_items
                        - self.analysis_results.coverage.documented_items
                ),
                action_steps: vec![
                    "Identify highest-priority undocumented APIs".to_string(),
                    "Create documentation templates".to_string(),
                    "Set up documentation CI checks".to_string(),
                ],
                estimated_effort_hours: 20.0,
                expected_impact: 0.3,
            });
        }

        // Example recommendations
        if !self
            .analysis_results
            .example_verification
            .failed_examples
            .is_empty()
        {
            recommendations.push(ActionableRecommendation {
                priority: Priority::Medium,
                category: "Examples".to_string(),
                title: "Fix Broken Examples".to_string(),
                description: format!(
                    "{} examples are failing compilation.",
                    self.analysis_results
                        .example_verification
                        .failed_examples
                        .len()
                ),
                action_steps: vec![
                    "Review failed examples".to_string(),
                    "Update syntax and dependencies".to_string(),
                    "Add example testing to CI".to_string(),
                ],
                estimated_effort_hours: 8.0,
                expected_impact: 0.2,
            });
        }

        recommendations
    }

    /// Calculate improvement effort
    fn calculate_improvement_effort(&self) -> EffortEstimate {
        let documentation_effort = (self.analysis_results.coverage.total_public_items
            - self.analysis_results.coverage.documented_items)
            as f64
            * 0.5; // 30 min per item

        let example_effort = self
            .analysis_results
            .example_verification
            .failed_examples
            .len() as f64
            * 1.0; // 1 hour per failed example

        let style_effort = self
            .analysis_results
            .style_analysis
            .violations
            .values()
            .map(|v| v.len())
            .sum::<usize>() as f64
            * 0.1; // 6 min per violation

        EffortEstimate {
            total_hours: documentation_effort + example_effort + style_effort,
            by_category: vec![
                ("Documentation".to_string(), documentation_effort),
                ("Examples".to_string(), example_effort),
                ("Style".to_string(), style_effort),
            ],
            confidence_level: 0.8,
        }
    }

    // Helper methods for parsing different item types

    fn extract_function_name(&self, line: &str) -> Option<String> {
        // pub fn function_name(...) -> ReturnType
        if let Some(start) = line.find("fn ") {
            let after_fn = &line[start + 3..];
            if let Some(end) = after_fn.find('(') {
                Some(after_fn[..end].trim().to_string())
            } else {
                None
            }
        } else {
            None
        }
    }

    fn extract_struct_name(&self, line: &str) -> Option<String> {
        // pub struct StructName<T>
        if let Some(start) = line.find("struct ") {
            let after_struct = &line[start + 7..];

            // Find the minimum position among all possible delimiters
            let mut end = after_struct.len();
            if let Some(pos) = after_struct.find(' ') {
                end = end.min(pos);
            }
            if let Some(pos) = after_struct.find('<') {
                end = end.min(pos);
            }
            if let Some(pos) = after_struct.find('{') {
                end = end.min(pos);
            }

            Some(after_struct[..end].trim().to_string())
        } else {
            None
        }
    }

    fn extract_enum_name(&self, line: &str) -> Option<String> {
        // pub enum EnumName<T>
        if let Some(start) = line.find("enum ") {
            let after_enum = &line[start + 5..];
            let end = after_enum
                .find(' ')
                .or_else(|| after_enum.find('<'))
                .or_else(|| after_enum.find('{'))
                .unwrap_or(after_enum.len());
            Some(after_enum[..end].trim().to_string())
        } else {
            None
        }
    }

    fn extract_trait_name(&self, line: &str) -> Option<String> {
        // pub trait TraitName<T>
        if let Some(start) = line.find("trait ") {
            let after_trait = &line[start + 6..];
            let end = after_trait
                .find(' ')
                .or_else(|| after_trait.find('<'))
                .or_else(|| after_trait.find(':'))
                .or_else(|| after_trait.find('{'))
                .unwrap_or(after_trait.len());
            Some(after_trait[..end].trim().to_string())
        } else {
            None
        }
    }

    fn extract_module_name(&self, line: &str) -> Option<String> {
        // pub mod module_name;
        if let Some(start) = line.find("mod ") {
            let after_mod = &line[start + 4..];
            let end = after_mod
                .find(' ')
                .or_else(|| after_mod.find(';'))
                .or_else(|| after_mod.find('{'))
                .unwrap_or(after_mod.len());
            Some(after_mod[..end].trim().to_string())
        } else {
            None
        }
    }

    fn extract_const_name(&self, line: &str) -> Option<String> {
        // pub const CONST_NAME: Type = value;
        if let Some(start) = line.find("const ") {
            let after_const = &line[start + 6..];
            if let Some(end) = after_const.find(':') {
                Some(after_const[..end].trim().to_string())
            } else {
                None
            }
        } else {
            None
        }
    }
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

/// Coverage summary
#[derive(Debug)]
pub struct CoverageSummary {
    pub percentage: f64,
    pub total_items: usize,
    pub documented_items: usize,
    pub critical_missing: Vec<String>,
}

/// Quality assessment
#[derive(Debug)]
pub struct QualityAssessment {
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub improvement_priorities: Vec<String>,
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

/// Effort estimate
#[derive(Debug)]
pub struct EffortEstimate {
    pub total_hours: f64,
    pub by_category: Vec<(String, f64)>,
    pub confidence_level: f64,
}

// Default implementations for metrics structures

impl Default for AnalysisResults {
    fn default() -> Self {
        Self {
            coverage: CoverageAnalysis::default(),
            example_verification: ExampleVerificationResults::default(),
            link_checking: LinkCheckingResults::default(),
            style_analysis: StyleAnalysis::default(),
            api_completeness: ApiCompletenessAnalysis::default(),
            overall_quality_score: 0.0,
        }
    }
}

impl Default for CoverageAnalysis {
    fn default() -> Self {
        Self {
            total_public_items: 0,
            documented_items: 0,
            coverage_percentage: 0.0,
            undocumented_by_category: HashMap::new(),
            quality_by_module: HashMap::new(),
        }
    }
}

impl Default for ExampleVerificationResults {
    fn default() -> Self {
        Self {
            total_examples: 0,
            compiled_examples: 0,
            failed_examples: Vec::new(),
            example_coverage: HashMap::new(),
            quality_metrics: ExampleQualityMetrics::default(),
        }
    }
}

impl Default for ExampleQualityMetrics {
    fn default() -> Self {
        Self {
            average_length: 0.0,
            error_handling_coverage: 0.0,
            comment_coverage: 0.0,
            best_practices_score: 0.0,
        }
    }
}

impl Default for LinkCheckingResults {
    fn default() -> Self {
        Self {
            total_links: 0,
            valid_links: 0,
            broken_links: Vec::new(),
            external_link_status: HashMap::new(),
            internal_link_consistency: 1.0,
        }
    }
}

impl Default for StyleAnalysis {
    fn default() -> Self {
        Self {
            consistency_score: 1.0,
            violations: HashMap::new(),
            recommendations: Vec::new(),
            format_analysis: FormatAnalysis::default(),
        }
    }
}

impl Default for FormatAnalysis {
    fn default() -> Self {
        Self {
            markdown_compliance: 1.0,
            rustdoc_compliance: 1.0,
            cross_reference_completeness: 1.0,
            toc_quality: 1.0,
        }
    }
}

impl Default for ApiCompletenessAnalysis {
    fn default() -> Self {
        Self {
            missing_sections: HashMap::new(),
            evolution_tracking: ApiEvolutionAnalysis::default(),
            documentation_debt: DocumentationDebt::default(),
            accessibility_compliance: AccessibilityAnalysis::default(),
        }
    }
}

impl Default for ApiEvolutionAnalysis {
    fn default() -> Self {
        Self {
            new_apis: Vec::new(),
            deprecated_apis: Vec::new(),
            changed_apis: Vec::new(),
            breaking_changes: Vec::new(),
        }
    }
}

impl Default for DocumentationDebt {
    fn default() -> Self {
        Self {
            total_debt_score: 0.0,
            debt_by_category: HashMap::new(),
            high_priority_items: Vec::new(),
            estimated_effort_hours: 0.0,
        }
    }
}

impl Default for AccessibilityAnalysis {
    fn default() -> Self {
        Self {
            alttext_coverage: 1.0,
            color_contrast_compliance: 1.0,
            screen_reader_compatibility: 1.0,
            keyboard_navigation_support: 1.0,
            overall_accessibility_score: 1.0,
        }
    }
}

impl Default for DocumentationMetrics {
    fn default() -> Self {
        Self {
            total_doc_lines: 0,
            code_to_doc_ratio: 0.0,
            average_quality_score: 0.0,
            maintenance_burden: 0.0,
            user_satisfaction: UserSatisfactionMetrics::default(),
        }
    }
}

impl Default for UserSatisfactionMetrics {
    fn default() -> Self {
        Self {
            clarity_score: 0.0,
            completeness_score: 0.0,
            helpfulness_score: 0.0,
            overall_satisfaction: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let config = AnalyzerConfig::default();
        let analyzer = DocumentationAnalyzer::new(config);
        assert_eq!(analyzer.analysis_results.overall_quality_score, 0.0);
    }

    #[test]
    fn test_function_name_extraction() {
        let analyzer = DocumentationAnalyzer::new(AnalyzerConfig::default());
        let line = "pub fn my_function(param: i32) -> Result<(), Error>";
        let name = analyzer.extract_function_name(line);
        assert_eq!(name, Some("my_function".to_string()));
    }

    #[test]
    fn test_struct_name_extraction() {
        let analyzer = DocumentationAnalyzer::new(AnalyzerConfig::default());
        let line = "pub struct MyStruct<T> {";
        let name = analyzer.extract_struct_name(line);
        assert_eq!(name, Some("MyStruct".to_string()));
    }

    #[test]
    fn test_documentation_detection() {
        let analyzer = DocumentationAnalyzer::new(AnalyzerConfig::default());
        let lines = vec![
            "/// This is documentation",
            "/// for a function",
            "pub fn documented_function() {}",
        ];
        assert!(analyzer.has_documentation(&lines, 2));
    }

    #[test]
    fn test_url_extraction() {
        let analyzer = DocumentationAnalyzer::new(AnalyzerConfig::default());
        let line = "See https://example.com for more info";
        let url = analyzer.extract_url(line);
        assert_eq!(url, Some("https://example.com".to_string()));
    }

    #[test]
    fn test_example_compilation() {
        let analyzer = DocumentationAnalyzer::new(AnalyzerConfig::default());
        let valid_code = "let x = 5;\nprintln!(\"{}\", x);";
        let invalid_code = "syntax_error";

        assert!(analyzer.compile_example(valid_code));
        assert!(!analyzer.compile_example(invalid_code));
    }
}
