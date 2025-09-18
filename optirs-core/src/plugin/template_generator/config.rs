// Template generator configuration types
//
// This module contains all configuration-related structures and enums
// for the template generator system.

use serde::{Deserialize, Serialize};

/// Template generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateGeneratorConfig {
    /// Target Rust version
    pub rust_version: String,
    /// Include CI/CD templates
    pub include_cicd: bool,
    /// Include documentation templates
    pub include_docs: bool,
    /// Include benchmark templates
    pub include_benchmarks: bool,
    /// Include example templates
    pub include_examples: bool,
    /// Include GPU support templates
    pub include_gpu: bool,
    /// Include distributed training templates
    pub include_distributed: bool,
    /// Code style preferences
    pub code_style: CodeStyle,
    /// License type
    pub license: LicenseType,
    /// Testing framework
    pub testing_framework: TestingFramework,
}

/// Code style configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeStyle {
    /// Indentation type
    pub indentation: IndentationType,
    /// Import organization style
    pub import_style: ImportStyle,
    /// Documentation style
    pub doc_style: DocStyle,
    /// Maximum line length
    pub max_line_length: usize,
    /// Use trailing commas
    pub trailing_commas: bool,
}

/// Indentation type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IndentationType {
    Spaces(u8),
    Tabs,
}

/// Import organization style
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImportStyle {
    Grouped,
    Alphabetical,
    Length,
    Mixed,
}

/// Documentation style
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DocStyle {
    Standard,
    Comprehensive,
    Minimal,
}

/// License type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LicenseType {
    MIT,
    Apache2,
    GPL3,
    BSD3,
    Proprietary,
    Custom,
}

/// Testing framework
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TestingFramework {
    Standard,
    Proptest,
    Criterion,
    Custom,
}

impl Default for TemplateGeneratorConfig {
    fn default() -> Self {
        Self {
            rust_version: "1.70.0".to_string(),
            include_cicd: true,
            include_docs: true,
            include_benchmarks: true,
            include_examples: true,
            include_gpu: false,
            include_distributed: false,
            code_style: CodeStyle::default(),
            license: LicenseType::MIT,
            testing_framework: TestingFramework::Standard,
        }
    }
}

impl Default for CodeStyle {
    fn default() -> Self {
        Self {
            indentation: IndentationType::Spaces(4),
            import_style: ImportStyle::Grouped,
            doc_style: DocStyle::Standard,
            max_line_length: 100,
            trailing_commas: true,
        }
    }
}