//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::documentationanalyzer_type::DocumentationAnalyzer;
use super::types::AnalyzerConfig;

#[cfg(test)]
pub(super) mod tests {
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
    fn test_url_extraction_trims_trailing_punctuation() {
        let analyzer = DocumentationAnalyzer::new(AnalyzerConfig::default());
        assert_eq!(
            analyzer.extract_url("see https://example.com, next"),
            Some("https://example.com".to_string())
        );
        assert_eq!(
            analyzer.extract_url("(https://example.com/path)."),
            Some("https://example.com/path".to_string())
        );
    }

    #[test]
    fn test_validate_url_is_syntactic_not_always_true() {
        let analyzer = DocumentationAnalyzer::new(AnalyzerConfig::default());
        assert!(analyzer.validate_url("https://example.com/path"));
        assert!(analyzer.validate_url("http://sub.example.org"));
        // No scheme, whitespace, or bare host must be rejected -- the old
        // implementation returned true for everything.
        assert!(!analyzer.validate_url("example.com"));
        assert!(!analyzer.validate_url("https://"));
        assert!(!analyzer.validate_url("not a url"));
        assert!(!analyzer.validate_url("ftp://example.com"));
    }

    #[test]
    fn test_example_compilation() {
        let analyzer = DocumentationAnalyzer::new(AnalyzerConfig::default());
        let valid_code = "let x = 5;\nprintln!(\"{}\", x);";
        // Unbalanced/broken syntax that fails both the rustc check and the
        // structural fallback used when rustc is unavailable.
        let invalid_code = "let x = (;";

        assert!(analyzer.compile_example(valid_code));
        assert!(!analyzer.compile_example(invalid_code));
        assert!(!analyzer.compile_example("   "));
    }
}
