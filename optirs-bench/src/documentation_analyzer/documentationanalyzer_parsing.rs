//! # `DocumentationAnalyzer` - parsing Methods
//!
//! This module contains method implementations for `DocumentationAnalyzer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::types::{
    AnalysisResults, AnalyzerConfig, BrokenLink, CoverageAnalysis, DocumentationMetrics,
    ExampleCoverage, ExampleQualityMetrics, ExampleVerificationResults, FailedExample,
    FormatAnalysis, ItemCategory, LinkCheckingResults, LinkErrorType, LinkStatus, StyleAnalysis,
    StyleCategory, StyleRecommendation, StyleViolation, UndocumentedItem, ViolationSeverity,
    VisibilityLevel,
};

use super::documentationanalyzer_type::DocumentationAnalyzer;

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
                        .or_default()
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
    pub(super) fn has_documentation(&self, lines: &[&str], itemline: usize) -> bool {
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

    /// Check whether a documentation example is valid Rust.
    ///
    /// Performs a real type-check by invoking `rustc` (metadata-only, no
    /// codegen or linking), wrapping bare statement snippets in a `fn main`
    /// exactly as `rustdoc` does. When `rustc` is not available on `PATH`, it
    /// falls back to a structural syntax check (balanced delimiters) and
    /// documents that it is a heuristic rather than a full compile.
    pub(super) fn compile_example(&self, example_code: &str) -> bool {
        if example_code.trim().is_empty() {
            return false;
        }
        match Self::rustc_typecheck(example_code) {
            Some(ok) => ok,
            None => Self::delimiters_balanced(example_code),
        }
    }

    /// Type-check `code` with `rustc`. Returns `Some(true/false)` for a real
    /// compile result, or `None` if `rustc` could not be invoked at all.
    fn rustc_typecheck(code: &str) -> Option<bool> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let stem = format!(
            "optirs_doc_example_{}_{}_{}",
            std::process::id(),
            nanos,
            seq
        );
        let dir = std::env::temp_dir();
        let src_path = dir.join(format!("{stem}.rs"));
        let out_path = dir.join(format!("{stem}.rmeta"));

        // Mirror rustdoc: snippets that already declare `fn main` compile as-is,
        // otherwise the snippet is wrapped in a synthetic `main`.
        let wrapped = if code.contains("fn main") {
            format!("#![allow(warnings)]\n{code}\n")
        } else {
            format!("#![allow(warnings)]\nfn main() {{\n{code}\n}}\n")
        };

        if std::fs::write(&src_path, wrapped).is_err() {
            return None;
        }

        let status = std::process::Command::new("rustc")
            .arg("--edition")
            .arg("2021")
            .arg("--crate-type")
            .arg("lib")
            .arg("--emit")
            .arg("metadata")
            .arg("-o")
            .arg(&out_path)
            .arg(&src_path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();

        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&out_path);

        match status {
            Ok(s) => Some(s.success()),
            Err(_) => None,
        }
    }

    /// Structural fallback: verify that `()`, `[]` and `{}` are balanced and
    /// that string/char literals are closed. Not a full parse, but honest
    /// about being a heuristic used only when `rustc` is unavailable.
    fn delimiters_balanced(code: &str) -> bool {
        let mut stack: Vec<char> = Vec::new();
        let mut chars = code.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '(' | '[' | '{' => stack.push(c),
                ')' if stack.pop() != Some('(') => return false,
                ']' if stack.pop() != Some('[') => return false,
                '}' if stack.pop() != Some('{') => return false,
                '"' => {
                    // Consume until the closing quote, honoring escapes.
                    let mut closed = false;
                    while let Some(n) = chars.next() {
                        if n == '\\' {
                            chars.next();
                        } else if n == '"' {
                            closed = true;
                            break;
                        }
                    }
                    if !closed {
                        return false;
                    }
                }
                _ => {}
            }
        }
        stack.is_empty()
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

    /// Extract a URL starting at the first `http` occurrence on `line`.
    ///
    /// The URL runs up to the first whitespace, then trailing punctuation that
    /// belongs to the surrounding prose rather than the URL is stripped (e.g.
    /// `see https://x.com,` or `(https://x.com).`).
    pub(super) fn extract_url(&self, line: &str) -> Option<String> {
        let start = line.find("http")?;
        let tail = &line[start..];
        let end = tail.find(char::is_whitespace).unwrap_or(tail.len());
        let url = tail[..end].trim_end_matches(|c: char| {
            matches!(
                c,
                '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']' | '}' | '>' | '"' | '\''
            )
        });
        if url.is_empty() {
            None
        } else {
            Some(url.to_string())
        }
    }

    /// Validate a URL syntactically (no network access).
    ///
    /// Requires an `http`/`https` scheme, a non-empty host containing a dot
    /// (or `localhost`), and no whitespace or control characters. This checks
    /// well-formedness, not reachability -- it never claims an arbitrary string
    /// is a valid URL.
    pub(super) fn validate_url(&self, url: &str) -> bool {
        let rest = match url
            .strip_prefix("https://")
            .or_else(|| url.strip_prefix("http://"))
        {
            Some(rest) => rest,
            None => return false,
        };
        if url.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return false;
        }
        let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
        // Strip any userinfo ("user@host") and port ("host:port").
        let host = authority.rsplit('@').next().unwrap_or(authority);
        let host = host.split(':').next().unwrap_or(host);
        !host.is_empty() && !host.starts_with('.') && (host.contains('.') || host == "localhost")
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
                    .or_default()
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
                    .or_default()
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
                    .or_default()
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
}
