// Advanced Plugin Template Generator
//
// This module provides comprehensive template generation capabilities for creating
// sophisticated optimizer plugins with advanced features, testing, documentation,
// and CI/CD integration.

use super::core::*;
use crate::error::{OptimError, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// Import all submodules
pub mod config;
pub mod templates;
pub mod registry;
pub mod generators;
pub mod validation;

// Re-export key types for convenience
pub use config::*;
pub use templates::*;
pub use registry::*;
pub use generators::*;
pub use validation::*;

/// Advanced template generator for plugin development
#[derive(Debug)]
pub struct AdvancedTemplateGenerator {
    /// Template configuration
    config: TemplateGeneratorConfig,
    /// Template registry
    template_registry: TemplateRegistry,
    /// Code generators
    generators: HashMap<GeneratorType, Box<dyn CodeGenerator>>,
    /// Validation engine
    validator: TemplateValidator,
}

impl AdvancedTemplateGenerator {
    /// Create a new template generator
    pub fn new(config: TemplateGeneratorConfig) -> Self {
        let mut generators: HashMap<GeneratorType, Box<dyn CodeGenerator>> = HashMap::new();

        // Register default generators
        generators.insert(GeneratorType::CorePlugin, Box::new(CorePluginGenerator::new()));
        generators.insert(GeneratorType::Documentation, Box::new(DocumentationGenerator));
        generators.insert(GeneratorType::Tests, Box::new(TestGenerator));

        Self {
            config,
            template_registry: TemplateRegistry::new(),
            generators,
            validator: TemplateValidator::new(),
        }
    }

    /// Generate a complete plugin from a template
    pub fn generate_plugin(&self, template_id: &str, output_path: &Path) -> Result<Vec<GeneratedFile>> {
        let template = self.template_registry
            .get_template(template_id)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Template '{}' not found", template_id)))?;

        // Validate template
        let validation_report = self.validator.validate(template);
        if !validation_report.is_valid() {
            return Err(OptimError::InvalidConfig(
                format!("Template validation failed: {:?}", validation_report.errors())
            ));
        }

        let mut all_files = Vec::new();

        // Generate files using all applicable generators
        for (generator_type, generator) in &self.generators {
            if generator.can_handle(template) {
                let files = generator.generate(template, &self.config)?;
                all_files.extend(files);
            }
        }

        Ok(all_files)
    }

    /// Register a new template
    pub fn register_template(&mut self, template: PluginTemplate) -> Result<()> {
        self.template_registry.register_template(template)
    }

    /// Add a custom code generator
    pub fn add_generator(&mut self, generator_type: GeneratorType, generator: Box<dyn CodeGenerator>) {
        self.generators.insert(generator_type, generator);
    }

    /// Get template registry statistics
    pub fn get_statistics(&self) -> RegistryStatistics {
        self.template_registry.get_statistics()
    }

    /// List available templates
    pub fn list_templates(&self) -> Vec<&String> {
        self.template_registry.list_template_ids()
    }

    /// Search templates by category
    pub fn get_templates_by_category(&self, category: TemplateCategory) -> Option<&Vec<PluginTemplate>> {
        self.template_registry.get_templates_by_category(category)
    }

    /// Create a basic optimizer template
    pub fn create_basic_optimizer_template(
        name: String,
        description: String,
        author: String,
    ) -> PluginTemplate {
        let metadata = TemplateMetadata::new(
            format!("basic-{}", name.to_lowercase().replace(' ', "-")),
            "0.1.0".to_string(),
            author,
            description.clone(),
        );

        let structure = EnhancedTemplateStructure {
            name,
            description,
            category: TemplateCategory::Basic,
            complexity: ComplexityLevel::Beginner,
            required_features: vec!["optimization".to_string()],
            optional_features: vec!["gpu".to_string(), "distributed".to_string()],
            file_structure: HashMap::new(),
            config_overrides: HashMap::new(),
        };

        PluginTemplate::new(metadata, structure)
            .with_dependency("optirs-core".to_string())
            .with_dependency("ndarray".to_string())
            .with_dependency("num-traits".to_string())
    }

    /// Write generated files to disk
    pub fn write_files(&self, files: Vec<GeneratedFile>, base_path: &Path) -> Result<()> {
        for file in files {
            let full_path = base_path.join(&file.path);

            // Create parent directories if they don't exist
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| OptimError::InvalidConfig(format!("Failed to create directory: {}", e)))?;
            }

            // Write file content
            std::fs::write(&full_path, &file.content)
                .map_err(|e| OptimError::InvalidConfig(format!("Failed to write file: {}", e)))?;

            // Set executable if needed
            if file.executable {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let metadata = std::fs::metadata(&full_path)
                        .map_err(|e| OptimError::InvalidConfig(format!("Failed to get file metadata: {}", e)))?;
                    let mut permissions = metadata.permissions();
                    permissions.set_mode(permissions.mode() | 0o755);
                    std::fs::set_permissions(&full_path, permissions)
                        .map_err(|e| OptimError::InvalidConfig(format!("Failed to set permissions: {}", e)))?;
                }
            }
        }

        Ok(())
    }
}

impl Default for AdvancedTemplateGenerator {
    fn default() -> Self {
        Self::new(TemplateGeneratorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_generator_creation() {
        let generator = AdvancedTemplateGenerator::default();
        assert_eq!(generator.list_templates().len(), 0);
    }

    #[test]
    fn test_basic_template_creation() {
        let template = AdvancedTemplateGenerator::create_basic_optimizer_template(
            "TestOptimizer".to_string(),
            "A test optimizer for unit testing".to_string(),
            "Test Author".to_string(),
        );

        assert_eq!(template.structure.name, "TestOptimizer");
        assert_eq!(template.structure.category, TemplateCategory::Basic);
        assert_eq!(template.structure.complexity, ComplexityLevel::Beginner);
    }
}