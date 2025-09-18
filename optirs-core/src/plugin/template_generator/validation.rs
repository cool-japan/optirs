// Template validation system
//
// This module provides validation capabilities for plugin templates,
// ensuring they meet quality and compatibility requirements.

use super::templates::*;
use super::config::*;
use crate::error::{OptimError, Result};

/// Template validator for ensuring template quality and compatibility
#[derive(Debug, Default)]
pub struct TemplateValidator {
    /// Validation rules
    rules: Vec<ValidationRule>,
}

/// Validation rule definition
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Validation function
    pub validator: fn(&PluginTemplate) -> ValidationResult,
    /// Whether this rule is mandatory
    pub mandatory: bool,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub passed: bool,
    /// Validation message
    pub message: String,
    /// Severity level
    pub severity: ValidationSeverity,
}

/// Validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Complete validation report
#[derive(Debug)]
pub struct ValidationReport {
    /// Template ID that was validated
    pub template_id: String,
    /// Overall validation result
    pub overall_result: bool,
    /// Individual rule results
    pub rule_results: Vec<ValidationResult>,
    /// Validation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl TemplateValidator {
    /// Create a new validator with default rules
    pub fn new() -> Self {
        let mut validator = Self {
            rules: Vec::new(),
        };

        validator.add_default_rules();
        validator
    }

    /// Add a validation rule
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }

    /// Validate a template against all rules
    pub fn validate(&self, template: &PluginTemplate) -> ValidationReport {
        let mut rule_results = Vec::new();
        let mut overall_result = true;

        for rule in &self.rules {
            let result = (rule.validator)(template);

            if rule.mandatory && !result.passed {
                overall_result = false;
            }

            rule_results.push(result);
        }

        ValidationReport {
            template_id: template.metadata.id.clone(),
            overall_result,
            rule_results,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Add default validation rules
    fn add_default_rules(&mut self) {
        // Template ID validation
        self.add_rule(ValidationRule {
            name: "template_id".to_string(),
            description: "Template ID must be non-empty and valid".to_string(),
            validator: |template| {
                if template.metadata.id.is_empty() {
                    ValidationResult {
                        passed: false,
                        message: "Template ID cannot be empty".to_string(),
                        severity: ValidationSeverity::Error,
                    }
                } else {
                    ValidationResult {
                        passed: true,
                        message: "Template ID is valid".to_string(),
                        severity: ValidationSeverity::Info,
                    }
                }
            },
            mandatory: true,
        });

        // Template name validation
        self.add_rule(ValidationRule {
            name: "template_name".to_string(),
            description: "Template name must be non-empty".to_string(),
            validator: |template| {
                if template.structure.name.is_empty() {
                    ValidationResult {
                        passed: false,
                        message: "Template name cannot be empty".to_string(),
                        severity: ValidationSeverity::Error,
                    }
                } else {
                    ValidationResult {
                        passed: true,
                        message: "Template name is valid".to_string(),
                        severity: ValidationSeverity::Info,
                    }
                }
            },
            mandatory: true,
        });

        // Description validation
        self.add_rule(ValidationRule {
            name: "description".to_string(),
            description: "Template should have a meaningful description".to_string(),
            validator: |template| {
                if template.structure.description.len() < 20 {
                    ValidationResult {
                        passed: false,
                        message: "Template description should be at least 20 characters".to_string(),
                        severity: ValidationSeverity::Warning,
                    }
                } else {
                    ValidationResult {
                        passed: true,
                        message: "Template description is adequate".to_string(),
                        severity: ValidationSeverity::Info,
                    }
                }
            },
            mandatory: false,
        });

        // Author validation
        self.add_rule(ValidationRule {
            name: "author".to_string(),
            description: "Template should have author information".to_string(),
            validator: |template| {
                if template.metadata.author.is_empty() {
                    ValidationResult {
                        passed: false,
                        message: "Template should have author information".to_string(),
                        severity: ValidationSeverity::Warning,
                    }
                } else {
                    ValidationResult {
                        passed: true,
                        message: "Template has author information".to_string(),
                        severity: ValidationSeverity::Info,
                    }
                }
            },
            mandatory: false,
        });
    }
}

impl ValidationReport {
    /// Get all errors from the validation report
    pub fn errors(&self) -> Vec<&ValidationResult> {
        self.rule_results
            .iter()
            .filter(|r| !r.passed && matches!(r.severity, ValidationSeverity::Error | ValidationSeverity::Critical))
            .collect()
    }

    /// Get all warnings from the validation report
    pub fn warnings(&self) -> Vec<&ValidationResult> {
        self.rule_results
            .iter()
            .filter(|r| matches!(r.severity, ValidationSeverity::Warning))
            .collect()
    }

    /// Check if validation passed completely
    pub fn is_valid(&self) -> bool {
        self.overall_result
    }
}