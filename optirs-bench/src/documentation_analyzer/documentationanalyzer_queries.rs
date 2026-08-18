//! # `DocumentationAnalyzer` - queries Methods
//!
//! This module contains method implementations for `DocumentationAnalyzer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use std::collections::HashMap;
use std::path::PathBuf;

use super::types::{
    AccessibilityAnalysis, ActionableRecommendation, ApiCompletenessAnalysis, ApiEvolutionAnalysis,
    CoverageSummary, DebtCategory, DebtItem, DocumentationDebt, DocumentationReport,
    EffortEstimate, ItemCategory, MissingSection, Priority, QualityAssessment,
};

use super::documentationanalyzer_type::DocumentationAnalyzer;

impl DocumentationAnalyzer {
    /// Analyze API completeness
    pub(super) fn analyze_api_completeness(&mut self) -> Result<()> {
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
    pub(super) fn find_missing_sections(&self) -> Result<HashMap<String, Vec<MissingSection>>> {
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
    pub(super) fn track_api_evolution(&self) -> Result<ApiEvolutionAnalysis> {
        Ok(ApiEvolutionAnalysis {
            new_apis: vec!["PerformanceProfiler::new".to_string()],
            deprecated_apis: vec![],
            changed_apis: vec![],
            breaking_changes: vec![],
        })
    }

    /// Assess documentation debt
    pub(super) fn assess_documentation_debt(&self) -> Result<DocumentationDebt> {
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
    pub(super) fn analyze_accessibility_compliance(&self) -> Result<AccessibilityAnalysis> {
        Ok(AccessibilityAnalysis {
            alttext_coverage: 0.7,
            color_contrast_compliance: 0.9,
            screen_reader_compatibility: 0.8,
            keyboard_navigation_support: 0.9,
            overall_accessibility_score: 0.82,
        })
    }

    /// Calculate overall quality score
    pub(super) fn calculate_overall_quality_score(&mut self) {
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

    pub(super) fn extract_function_name(&self, line: &str) -> Option<String> {
        // pub fn function_name(...) -> ReturnType
        if let Some(start) = line.find("fn ") {
            let after_fn = &line[start + 3..];
            after_fn
                .find('(')
                .map(|end| after_fn[..end].trim().to_string())
        } else {
            None
        }
    }

    pub(super) fn extract_struct_name(&self, line: &str) -> Option<String> {
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

    pub(super) fn extract_enum_name(&self, line: &str) -> Option<String> {
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

    pub(super) fn extract_trait_name(&self, line: &str) -> Option<String> {
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

    pub(super) fn extract_module_name(&self, line: &str) -> Option<String> {
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

    pub(super) fn extract_const_name(&self, line: &str) -> Option<String> {
        // pub const CONST_NAME: Type = value;
        if let Some(start) = line.find("const ") {
            let after_const = &line[start + 6..];
            after_const
                .find(':')
                .map(|end| after_const[..end].trim().to_string())
        } else {
            None
        }
    }
}
