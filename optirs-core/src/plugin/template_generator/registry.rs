// Template registry and management
//
// This module provides template registry functionality for storing,
// organizing, and retrieving plugin templates.

use super::templates::*;
use crate::error::{OptimError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Template registry for managing plugin templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateRegistry {
    /// Templates organized by category
    templates: HashMap<TemplateCategory, Vec<PluginTemplate>>,
    /// Template lookup by ID
    template_lookup: HashMap<String, (TemplateCategory, usize)>,
    /// Registry metadata
    metadata: RegistryMetadata,
}

/// Registry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryMetadata {
    /// Registry version
    pub version: String,
    /// Last update timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
    /// Total number of templates
    pub template_count: usize,
    /// Registry description
    pub description: String,
}

impl TemplateRegistry {
    /// Create a new template registry
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            template_lookup: HashMap::new(),
            metadata: RegistryMetadata {
                version: "1.0.0".to_string(),
                last_updated: chrono::Utc::now(),
                template_count: 0,
                description: "OptiRS Plugin Template Registry".to_string(),
            },
        }
    }

    /// Register a new template
    pub fn register_template(&mut self, template: PluginTemplate) -> Result<()> {
        let category = template.structure.category;
        let template_id = template.metadata.id.clone();

        // Check for duplicate IDs
        if self.template_lookup.contains_key(&template_id) {
            return Err(OptimError::InvalidConfig(
                format!("Template with ID '{}' already exists", template_id)
            ));
        }

        // Add to category list
        let templates = self.templates.entry(category).or_insert_with(Vec::new);
        let index = templates.len();
        templates.push(template);

        // Update lookup
        self.template_lookup.insert(template_id, (category, index));

        // Update metadata
        self.metadata.template_count += 1;
        self.metadata.last_updated = chrono::Utc::now();

        Ok(())
    }

    /// Get a template by ID
    pub fn get_template(&self, template_id: &str) -> Option<&PluginTemplate> {
        self.template_lookup.get(template_id)
            .and_then(|(category, index)| {
                self.templates.get(category)
                    .and_then(|templates| templates.get(*index))
            })
    }

    /// Get all templates in a category
    pub fn get_templates_by_category(&self, category: TemplateCategory) -> Option<&Vec<PluginTemplate>> {
        self.templates.get(&category)
    }

    /// Get templates by complexity level
    pub fn get_templates_by_complexity(&self, complexity: ComplexityLevel) -> Vec<&PluginTemplate> {
        self.templates
            .values()
            .flatten()
            .filter(|t| t.structure.complexity == complexity)
            .collect()
    }

    /// Search templates by tag
    pub fn search_by_tag(&self, tag: &str) -> Vec<&PluginTemplate> {
        self.templates
            .values()
            .flatten()
            .filter(|t| t.metadata.tags.contains(&tag.to_string()))
            .collect()
    }

    /// List all available template IDs
    pub fn list_template_ids(&self) -> Vec<&String> {
        self.template_lookup.keys().collect()
    }

    /// Get registry statistics
    pub fn get_statistics(&self) -> RegistryStatistics {
        let mut category_counts = HashMap::new();
        let mut complexity_counts = HashMap::new();

        for templates in self.templates.values() {
            for template in templates {
                *category_counts.entry(template.structure.category).or_insert(0) += 1;
                *complexity_counts.entry(template.structure.complexity).or_insert(0) += 1;
            }
        }

        RegistryStatistics {
            total_templates: self.metadata.template_count,
            category_counts,
            complexity_counts,
            last_updated: self.metadata.last_updated,
        }
    }

    /// Remove a template by ID
    pub fn remove_template(&mut self, template_id: &str) -> Result<PluginTemplate> {
        if let Some((category, index)) = self.template_lookup.remove(template_id) {
            if let Some(templates) = self.templates.get_mut(&category) {
                if index < templates.len() {
                    let removed_template = templates.remove(index);

                    // Update indices in lookup table
                    for (_, (cat, idx)) in self.template_lookup.iter_mut() {
                        if *cat == category && *idx > index {
                            *idx -= 1;
                        }
                    }

                    self.metadata.template_count -= 1;
                    self.metadata.last_updated = chrono::Utc::now();

                    return Ok(removed_template);
                }
            }
        }

        Err(OptimError::InvalidConfig(
            format!("Template '{}' not found", template_id)
        ))
    }

    /// Clear all templates
    pub fn clear(&mut self) {
        self.templates.clear();
        self.template_lookup.clear();
        self.metadata.template_count = 0;
        self.metadata.last_updated = chrono::Utc::now();
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStatistics {
    pub total_templates: usize,
    pub category_counts: HashMap<TemplateCategory, usize>,
    pub complexity_counts: HashMap<ComplexityLevel, usize>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}