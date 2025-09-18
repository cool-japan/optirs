// Template structures and metadata
//
// This module contains template-related structures, metadata, and parameter types
// for the plugin template system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Plugin template structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginTemplate {
    /// Template metadata
    pub metadata: TemplateMetadata,
    /// Template structure
    pub structure: EnhancedTemplateStructure,
    /// Template parameters
    pub parameters: Vec<TemplateParameter>,
    /// Template content
    pub content: String,
    /// Dependencies
    pub dependencies: Vec<String>,
    /// Features
    pub features: Vec<String>,
}

/// Template categories
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TemplateCategory {
    Basic,
    Advanced,
    Specialized,
    Research,
    Production,
    Experimental,
    Educational,
    Performance,
    Distributed,
    Neuromorphic,
    Custom,
}

/// Complexity levels for templates
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComplexityLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Enhanced template structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedTemplateStructure {
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Template category
    pub category: TemplateCategory,
    /// Complexity level
    pub complexity: ComplexityLevel,
    /// Required features
    pub required_features: Vec<String>,
    /// Optional features
    pub optional_features: Vec<String>,
    /// File structure
    pub file_structure: HashMap<String, String>,
    /// Configuration overrides
    pub config_overrides: HashMap<String, serde_json::Value>,
}

/// Template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    /// Template ID
    pub id: String,
    /// Template version
    pub version: String,
    /// Author information
    pub author: String,
    /// Template description
    pub description: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modification timestamp
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Supported Rust versions
    pub rust_versions: Vec<String>,
    /// Template license
    pub license: String,
}

/// Template parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateParameter {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Default value
    pub default_value: Option<String>,
    /// Whether parameter is required
    pub required: bool,
    /// Parameter validation rules
    pub validation: Vec<ParameterValidation>,
}

/// Parameter types supported by templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Array(Box<ParameterType>),
    Object(HashMap<String, ParameterType>),
    Enum(Vec<String>),
    Custom(String),
}

/// Parameter validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValidation {
    MinLength(usize),
    MaxLength(usize),
    Pattern(String),
    Range { min: f64, max: f64 },
    OneOf(Vec<String>),
    Custom(String),
}

impl TemplateMetadata {
    pub fn new(id: String, version: String, author: String, description: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            id,
            version,
            author,
            description,
            created_at: now,
            modified_at: now,
            tags: Vec::new(),
            rust_versions: vec!["1.70.0".to_string()],
            license: "MIT".to_string(),
        }
    }
}

impl PluginTemplate {
    pub fn new(metadata: TemplateMetadata, structure: EnhancedTemplateStructure) -> Self {
        Self {
            metadata,
            structure,
            parameters: Vec::new(),
            content: String::new(),
            dependencies: Vec::new(),
            features: Vec::new(),
        }
    }

    pub fn with_parameter(mut self, parameter: TemplateParameter) -> Self {
        self.parameters.push(parameter);
        self
    }

    pub fn with_dependency(mut self, dependency: String) -> Self {
        self.dependencies.push(dependency);
        self
    }
}