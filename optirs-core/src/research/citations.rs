// Citation management and bibliographic tools
//
// This module provides comprehensive citation management, BibTeX parsing,
// and automated reference generation for academic publications.

use crate::error::{OptimError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Citation manager for handling bibliographic references
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationManager {
    /// Citation database
    pub citations: HashMap<String, Citation>,
    /// Citation styles
    pub styles: HashMap<String, CitationStyle>,
    /// Default citation style
    pub default_style: String,
    /// Citation groups/categories
    pub groups: HashMap<String, CitationGroup>,
    /// Import/export settings
    pub settings: CitationSettings,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
}

/// Individual citation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Citation key/identifier
    pub key: String,
    /// Publication type
    pub publication_type: PublicationType,
    /// Title
    pub title: String,
    /// Authors
    pub authors: Vec<Author>,
    /// Publication year
    pub year: Option<u32>,
    /// Journal/Conference/Publisher
    pub venue: Option<String>,
    /// Volume number
    pub volume: Option<String>,
    /// Issue/Number
    pub issue: Option<String>,
    /// Page numbers
    pub pages: Option<String>,
    /// DOI
    pub doi: Option<String>,
    /// URL
    pub url: Option<String>,
    /// Abstract
    pub abstracttext: Option<String>,
    /// Keywords
    pub keywords: Vec<String>,
    /// Notes
    pub notes: Option<String>,
    /// Custom fields
    pub custom_fields: HashMap<String, String>,
    /// File attachments
    pub attachments: Vec<String>,
    /// Citation groups
    pub groups: Vec<String>,
    /// Import source
    pub import_source: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
}

/// Publication types for citations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PublicationType {
    /// Journal article
    Article,
    /// Conference paper
    InProceedings,
    /// Book
    Book,
    /// Book chapter
    InCollection,
    /// PhD thesis
    PhDThesis,
    /// Master's thesis
    MastersThesis,
    /// Technical report
    TechReport,
    /// Manual
    Manual,
    /// Miscellaneous
    Misc,
    /// Unpublished work
    Unpublished,
    /// Preprint
    Preprint,
    /// Patent
    Patent,
    /// Software
    Software,
    /// Dataset
    Dataset,
}

/// Author information for citations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Author {
    /// First name
    pub first_name: String,
    /// Last name
    pub last_name: String,
    /// Middle name/initial
    pub middle_name: Option<String>,
    /// Name suffix (Jr., Sr., etc.)
    pub suffix: Option<String>,
    /// ORCID identifier
    pub orcid: Option<String>,
    /// Author affiliation
    pub affiliation: Option<String>,
}

/// Citation style definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationStyle {
    /// Style name
    pub name: String,
    /// Style description
    pub description: String,
    /// In-text citation format
    pub intext_format: InTextFormat,
    /// Bibliography format
    pub bibliography_format: BibliographyFormat,
    /// Formatting rules
    pub formatting_rules: FormattingRules,
    /// Sorting rules
    pub sorting_rules: SortingRules,
}

/// In-text citation formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum InTextFormat {
    /// Author-year format: (Smith, 2023)
    AuthorYear,
    /// Numbered format: [1]
    Numbered,
    /// Superscript format: ¹
    Superscript,
    /// Author-number format: Smith [1]
    AuthorNumber,
    /// Footnote format
    Footnote,
}

/// Bibliography formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BibliographyFormat {
    /// Entry separator
    pub entry_separator: String,
    /// Field separators
    pub field_separators: HashMap<String, String>,
    /// Name formatting
    pub name_format: NameFormat,
    /// Title formatting
    pub title_format: TitleFormat,
    /// Date formatting
    pub date_format: DateFormat,
    /// Punctuation rules
    pub punctuation: PunctuationRules,
}

/// Name formatting options
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NameFormat {
    /// Last, First Middle
    LastFirstMiddle,
    /// First Middle Last
    FirstMiddleLast,
    /// Last, F. M.
    LastFirstInitial,
    /// F. M. Last
    FirstInitialLast,
    /// Last, F.M.
    LastFirstInitialNoSpace,
}

/// Title formatting options
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TitleFormat {
    /// Title Case
    TitleCase,
    /// Sentence case
    SentenceCase,
    /// UPPERCASE
    Uppercase,
    /// lowercase
    Lowercase,
    /// As entered
    AsEntered,
}

/// Date formatting options
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DateFormat {
    /// 2023
    Year,
    /// December 2023
    MonthYear,
    /// Dec. 2023
    MonthAbbrevYear,
    /// December 15, 2023
    FullDate,
    /// 2023-12-15
    ISODate,
}

/// Punctuation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PunctuationRules {
    /// Use periods after abbreviations
    pub periods_after_abbreviations: bool,
    /// Use commas between fields
    pub commas_between_fields: bool,
    /// Use parentheses around year
    pub parentheses_around_year: bool,
    /// Quote titles
    pub quote_titles: bool,
    /// Italicize journal names
    pub italicize_journals: bool,
}

/// Formatting rules for citations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattingRules {
    /// Maximum authors to show
    pub max_authors: Option<usize>,
    /// Text to use for "et al."
    pub et_altext: String,
    /// Minimum authors before using et al.
    pub et_al_threshold: usize,
    /// Use title case for titles
    pub title_case: bool,
    /// Abbreviate journal names
    pub abbreviate_journals: bool,
    /// Include DOI
    pub include_doi: bool,
    /// Include URL
    pub include_url: bool,
}

/// Sorting rules for bibliography
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortingRules {
    /// Primary sort field
    pub primary_sort: SortField,
    /// Secondary sort field
    pub secondary_sort: Option<SortField>,
    /// Sort direction
    pub sort_direction: SortDirection,
    /// Group by type
    pub group_by_type: bool,
}

/// Sort fields
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SortField {
    /// Author last name
    Author,
    /// Publication year
    Year,
    /// Title
    Title,
    /// Journal/venue
    Venue,
    /// Citation key
    Key,
    /// Date added
    DateAdded,
}

/// Sort direction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SortDirection {
    /// Ascending order
    Ascending,
    /// Descending order
    Descending,
}

/// Citation group for organizing references
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationGroup {
    /// Group name
    pub name: String,
    /// Group description
    pub description: String,
    /// Group color (for UI)
    pub color: Option<String>,
    /// Citation keys in this group
    pub citation_keys: Vec<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Citation manager settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationSettings {
    /// Auto-generate keys
    pub auto_generate_keys: bool,
    /// Key generation pattern
    pub key_pattern: String,
    /// Auto-import from DOI
    pub auto_import_doi: bool,
    /// Auto-import from URL
    pub auto_import_url: bool,
    /// Duplicate detection
    pub duplicate_detection: bool,
    /// Backup settings
    pub backup_enabled: bool,
    /// Export formats
    pub export_formats: Vec<ExportFormat>,
}

/// Export formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExportFormat {
    /// BibTeX format
    BibTeX,
    /// RIS format
    RIS,
    /// EndNote XML
    EndNote,
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// Word bibliography
    Word,
}

/// BibTeX parser and exporter
#[derive(Debug)]
pub struct BibTeXProcessor {
    /// Parser settings
    settings: BibTeXSettings,
}

/// BibTeX processing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BibTeXSettings {
    /// Preserve case in titles
    pub preserve_case: bool,
    /// Convert to UTF-8
    pub utf8_conversion: bool,
    /// Clean up formatting
    pub cleanup_formatting: bool,
    /// Validate entries
    pub validate_entries: bool,
}

/// Citation search and discovery
#[derive(Debug)]
pub struct CitationDiscovery {
    /// Search engines configuration
    search_engines: Vec<SearchEngine>,
    /// API keys for services
    api_keys: HashMap<String, String>,
}

/// Search engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchEngine {
    /// Engine name
    pub name: String,
    /// API endpoint
    pub endpoint: String,
    /// Rate limit (requests per second)
    pub rate_limit: f64,
    /// Supported query types
    pub query_types: Vec<QueryType>,
}

/// Query types for citation search
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum QueryType {
    /// DOI lookup
    DOI,
    /// Title search
    Title,
    /// Author search
    Author,
    /// ArXiv ID
    ArXiv,
    /// PubMed ID
    PubMed,
    /// ISBN
    ISBN,
    /// Free text search
    FreeText,
}

/// Citation network analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationNetwork {
    /// Citations in the network
    pub citations: Vec<String>,
    /// Citation relationships
    pub relationships: Vec<CitationRelationship>,
    /// Network metrics
    pub metrics: NetworkMetrics,
}

/// Citation relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationRelationship {
    /// Citing paper
    pub citing: String,
    /// Cited paper
    pub cited: String,
    /// Relationship type
    pub relationship_type: RelationshipType,
    /// Relationship strength
    pub strength: f64,
}

/// Relationship types between citations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationshipType {
    /// Direct citation
    DirectCitation,
    /// Co-citation (cited together)
    CoCitation,
    /// Bibliographic coupling (share references)
    BibliographicCoupling,
    /// Same author
    SameAuthor,
    /// Same venue
    SameVenue,
    /// Similar topic
    SimilarTopic,
}

/// Network analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Total nodes (papers)
    pub total_nodes: usize,
    /// Total edges (relationships)
    pub total_edges: usize,
    /// Network density
    pub density: f64,
    /// Average clustering coefficient
    pub clustering_coefficient: f64,
    /// Most cited papers
    pub most_cited: Vec<(String, usize)>,
    /// Most influential authors
    pub most_influential_authors: Vec<(String, f64)>,
}

impl CitationManager {
    /// Create a new citation manager
    pub fn new() -> Self {
        let mut styles = HashMap::new();
        styles.insert("APA".to_string(), Self::create_apa_style());
        styles.insert("IEEE".to_string(), Self::create_ieee_style());
        styles.insert("ACM".to_string(), Self::create_acm_style());

        Self {
            citations: HashMap::new(),
            styles,
            default_style: "APA".to_string(),
            groups: HashMap::new(),
            settings: CitationSettings::default(),
            modified_at: Utc::now(),
        }
    }

    /// Add a citation to the database
    pub fn add_citation(&mut self, citation: Citation) -> Result<()> {
        if self.citations.contains_key(&citation.key) {
            return Err(OptimError::InvalidConfig(format!(
                "Citation with key '{}' already exists",
                citation.key
            )));
        }

        self.citations.insert(citation.key.clone(), citation);
        self.modified_at = Utc::now();
        Ok(())
    }

    /// Get a citation by key
    pub fn get_citation(&self, key: &str) -> Option<&Citation> {
        self.citations.get(key)
    }

    /// Update an existing citation
    pub fn update_citation(&mut self, key: &str, citation: Citation) -> Result<()> {
        if !self.citations.contains_key(key) {
            return Err(OptimError::InvalidConfig(format!(
                "Citation with key '{}' not found",
                key
            )));
        }

        self.citations.insert(key.to_string(), citation);
        self.modified_at = Utc::now();
        Ok(())
    }

    /// Remove a citation
    pub fn remove_citation(&mut self, key: &str) -> Result<()> {
        if self.citations.remove(key).is_none() {
            return Err(OptimError::InvalidConfig(format!(
                "Citation with key '{}' not found",
                key
            )));
        }

        self.modified_at = Utc::now();
        Ok(())
    }

    /// Search citations by various criteria
    pub fn search_citations(&self, query: &str) -> Vec<&Citation> {
        let query_lower = query.to_lowercase();

        self.citations
            .values()
            .filter(|citation| {
                citation.title.to_lowercase().contains(&query_lower)
                    || citation.authors.iter().any(|author| {
                        author.last_name.to_lowercase().contains(&query_lower)
                            || author.first_name.to_lowercase().contains(&query_lower)
                    })
                    || citation
                        .keywords
                        .iter()
                        .any(|keyword| keyword.to_lowercase().contains(&query_lower))
                    || citation
                        .venue
                        .as_ref()
                        .map_or(false, |venue| venue.to_lowercase().contains(&query_lower))
            })
            .collect()
    }

    /// Generate formatted citation in specified style
    pub fn format_citation(&self, key: &str, style: Option<&str>) -> Result<String> {
        let citation = self
            .get_citation(key)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Citation '{}' not found", key)))?;

        let style_name = style.unwrap_or(&self.default_style);
        let citation_style = self.styles.get(style_name).ok_or_else(|| {
            OptimError::InvalidConfig(format!("Style '{}' not found", style_name))
        })?;

        self.format_citation_with_style(citation, citation_style)
    }

    /// Generate bibliography for multiple citations
    pub fn generate_bibliography(
        &self,
        citation_keys: &[String],
        style: Option<&str>,
    ) -> Result<String> {
        let style_name = style.unwrap_or(&self.default_style);
        let citation_style = self.styles.get(style_name).ok_or_else(|| {
            OptimError::InvalidConfig(format!("Style '{}' not found", style_name))
        })?;

        let mut citations: Vec<&Citation> = citation_keys
            .iter()
            .filter_map(|key| self.citations.get(key))
            .collect();

        // Sort citations according to style rules
        self.sort_citations(&mut citations, &citation_style.sorting_rules);

        let mut bibliography = String::new();
        for citation in citations {
            let formatted = self.format_citation_with_style(citation, citation_style)?;
            bibliography.push_str(&formatted);
            bibliography.push('\n');
        }

        Ok(bibliography)
    }

    /// Export citations to BibTeX format
    pub fn export_bibtex(&self, citation_keys: Option<&[String]>) -> String {
        let citations: Vec<&Citation> = if let Some(_keys) = citation_keys {
            _keys
                .iter()
                .filter_map(|key| self.citations.get(key))
                .collect()
        } else {
            self.citations.values().collect()
        };

        let mut bibtex = String::new();
        for citation in citations {
            bibtex.push_str(&self.citation_to_bibtex(citation));
            bibtex.push('\n');
        }

        bibtex
    }

    /// Import citations from BibTeX
    pub fn import_bibtex(&mut self, bibtex_content: &str) -> Result<usize> {
        let processor = BibTeXProcessor::new(BibTeXSettings::default());
        let citations = processor.parse_bibtex(bibtex_content)?;

        let mut imported_count = 0;
        for citation in citations {
            if !self.citations.contains_key(&citation.key) {
                self.citations.insert(citation.key.clone(), citation);
                imported_count += 1;
            }
        }

        self.modified_at = Utc::now();
        Ok(imported_count)
    }

    /// Create a citation group
    pub fn create_group(&mut self, name: &str, description: &str) -> String {
        let group_id = uuid::Uuid::new_v4().to_string();
        let group = CitationGroup {
            name: name.to_string(),
            description: description.to_string(),
            color: None,
            citation_keys: Vec::new(),
            created_at: Utc::now(),
        };

        self.groups.insert(group_id.clone(), group);
        group_id
    }

    /// Add citation to group
    pub fn add_to_group(&mut self, group_id: &str, citation_key: &str) -> Result<()> {
        let group = self
            .groups
            .get_mut(group_id)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Group '{}' not found", group_id)))?;

        if !group.citation_keys.contains(&citation_key.to_string()) {
            group.citation_keys.push(citation_key.to_string());
        }

        Ok(())
    }

    fn format_citation_with_style(
        &self,
        citation: &Citation,
        style: &CitationStyle,
    ) -> Result<String> {
        match style.intext_format {
            InTextFormat::AuthorYear => self.format_author_year(citation, style),
            InTextFormat::Numbered => self.format_numbered(citation, style),
            InTextFormat::Superscript => self.format_superscript(citation, style),
            InTextFormat::AuthorNumber => self.format_author_number(citation, style),
            InTextFormat::Footnote => self.format_footnote(citation, style),
        }
    }

    fn format_author_year(&self, citation: &Citation, style: &CitationStyle) -> Result<String> {
        let authors = self.format_authors(&citation.authors, &style.formatting_rules);
        let year = citation
            .year
            .map(|y| y.to_string())
            .unwrap_or_else(|| "n.d.".to_string());

        Ok(format!("({}, {})", authors, year))
    }

    fn format_numbered(&self, citation: &Citation, style: &CitationStyle) -> Result<String> {
        // In a real implementation, you'd need to assign numbers based on order
        Ok(format!("[{}]", 1)) // Placeholder
    }

    fn format_superscript(&self, citation: &Citation, style: &CitationStyle) -> Result<String> {
        Ok("¹".to_string()) // Placeholder
    }

    fn format_author_number(&self, citation: &Citation, style: &CitationStyle) -> Result<String> {
        let authors = self.format_authors(&citation.authors, &style.formatting_rules);
        Ok(format!("{} [1]", authors)) // Placeholder
    }

    fn format_footnote(&self, citation: &Citation, style: &CitationStyle) -> Result<String> {
        self.format_full_citation(citation, style)
    }

    fn format_full_citation(&self, citation: &Citation, style: &CitationStyle) -> Result<String> {
        let mut formatted = String::new();

        // Authors
        let authors = self.format_authors(&citation.authors, &style.formatting_rules);
        formatted.push_str(&authors);

        // Title
        let title = self.format_title(&citation.title, &style.bibliography_format.title_format);
        formatted.push_str(&format!(". {}.", title));

        // Venue
        if let Some(venue) = &citation.venue {
            let venue_formatted = if style.bibliography_format.punctuation.italicize_journals {
                format!(" *{}*", venue)
            } else {
                format!(" {venue}")
            };
            formatted.push_str(&venue_formatted);
        }

        // Year
        if let Some(year) = citation.year {
            if style
                .bibliography_format
                .punctuation
                .parentheses_around_year
            {
                formatted.push_str(&format!(" ({})", year));
            } else {
                formatted.push_str(&format!(" {year}"));
            }
        }

        // DOI
        if style.formatting_rules.include_doi {
            if let Some(doi) = &citation.doi {
                formatted.push_str(&format!(". DOI: {doi}"));
            }
        }

        Ok(formatted)
    }

    fn format_authors(&self, authors: &[Author], rules: &FormattingRules) -> String {
        if authors.is_empty() {
            return "Anonymous".to_string();
        }

        let max_authors = rules.max_authors.unwrap_or(authors.len());
        let display_authors = if authors.len() > max_authors && max_authors > 0 {
            &authors[..max_authors]
        } else {
            authors
        };

        let mut formatted_authors = Vec::new();
        for author in display_authors {
            let formatted = format!("{}, {}", author.last_name, author.first_name);
            formatted_authors.push(formatted);
        }

        let mut result = formatted_authors.join(", ");

        if authors.len() > max_authors {
            result.push_str(&format!(", {}", rules.et_altext));
        }

        result
    }

    fn format_title(&self, title: &str, format: &TitleFormat) -> String {
        match format {
            TitleFormat::TitleCase => self.to_title_case(title),
            TitleFormat::SentenceCase => self.to_sentence_case(title),
            TitleFormat::Uppercase => title.to_uppercase(),
            TitleFormat::Lowercase => title.to_lowercase(),
            TitleFormat::AsEntered => title.to_string(),
        }
    }

    fn to_title_case(&self, s: &str) -> String {
        s.split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => {
                        first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                    }
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn to_sentence_case(&self, s: &str) -> String {
        if s.is_empty() {
            return String::new();
        }

        let mut chars = s.chars();
        let first = chars.next().unwrap().to_uppercase().collect::<String>();
        first + &chars.as_str().to_lowercase()
    }

    fn sort_citations(&self, citations: &mut Vec<&Citation>, rules: &SortingRules) {
        citations.sort_by(|a, b| {
            let primary_cmp = self.compare_by_field(a, b, &rules.primary_sort);
            if primary_cmp == std::cmp::Ordering::Equal {
                if let Some(secondary) = &rules.secondary_sort {
                    self.compare_by_field(a, b, secondary)
                } else {
                    std::cmp::Ordering::Equal
                }
            } else {
                primary_cmp
            }
        });

        if rules.sort_direction == SortDirection::Descending {
            citations.reverse();
        }
    }

    fn compare_by_field(
        &self,
        a: &Citation,
        b: &Citation,
        field: &SortField,
    ) -> std::cmp::Ordering {
        match field {
            SortField::Author => {
                let a_author = a.authors.first().map(|au| au.last_name.as_str()).unwrap_or("");
                let b_author = b.authors.first().map(|au| au.last_name.as_str()).unwrap_or("");
                a_author.cmp(b_author)
            }
            SortField::Year => a.year.cmp(&b.year),
            SortField::Title => a.title.cmp(&b.title),
            SortField::Venue => a.venue.cmp(&b.venue),
            SortField::Key => a.key.cmp(&b.key),
            SortField::DateAdded => a.created_at.cmp(&b.created_at),
        }
    }

    fn citation_to_bibtex(&self, citation: &Citation) -> String {
        let mut bibtex = format!(
            "@{}{{{},\n",
            self.publication_type_to_bibtex(&citation.publication_type),
            citation.key
        );

        bibtex.push_str(&format!("  title = {{{}}},\n", citation.title));

        if !citation.authors.is_empty() {
            let authors = citation
                .authors
                .iter()
                .map(|a| format!("{} {}", a.first_name, a.last_name))
                .collect::<Vec<_>>()
                .join(" and ");
            bibtex.push_str(&format!("  author = {{{}}},\n", authors));
        }

        if let Some(year) = citation.year {
            bibtex.push_str(&format!("  year = {{{}}},\n", year));
        }

        if let Some(venue) = &citation.venue {
            let field_name = match citation.publication_type {
                PublicationType::Article => "journal",
                PublicationType::InProceedings => "booktitle",
                PublicationType::Book => "publisher",
                PublicationType::InCollection => "booktitle",
                PublicationType::PhDThesis => "school",
                PublicationType::MastersThesis => "school",
                PublicationType::TechReport => "institution",
                PublicationType::Manual => "organization",
                PublicationType::Misc => "howpublished",
                PublicationType::Unpublished => "note",
                PublicationType::Preprint => "archivePrefix",
                PublicationType::Patent => "assignee",
                PublicationType::Software => "url",
                PublicationType::Dataset => "url",
            };
            bibtex.push_str(&format!("  {} = {{{}}},\n", field_name, venue));
        }

        if let Some(volume) = &citation.volume {
            bibtex.push_str(&format!("  volume = {{{}}},\n", volume));
        }

        if let Some(pages) = &citation.pages {
            bibtex.push_str(&format!("  pages = {{{}}},\n", pages));
        }

        if let Some(doi) = &citation.doi {
            bibtex.push_str(&format!("  doi = {{{}}},\n", doi));
        }

        bibtex.push_str("}\n");
        bibtex
    }

    fn publication_type_to_bibtex(&self, pub_type: &PublicationType) -> &'static str {
        match pub_type {
            PublicationType::Article => "article",
            PublicationType::InProceedings => "inproceedings",
            PublicationType::Book => "book",
            PublicationType::InCollection => "incollection",
            PublicationType::PhDThesis => "phdthesis",
            PublicationType::MastersThesis => "mastersthesis",
            PublicationType::TechReport => "techreport",
            PublicationType::Manual => "manual",
            PublicationType::Misc => "misc",
            PublicationType::Unpublished => "unpublished",
            PublicationType::Preprint => "misc",
            PublicationType::Patent => "misc",
            PublicationType::Software => "misc",
            PublicationType::Dataset => "misc",
        }
    }

    fn create_apa_style() -> CitationStyle {
        CitationStyle {
            name: "APA".to_string(),
            description: "American Psychological Association style".to_string(),
            intext_format: InTextFormat::AuthorYear,
            bibliography_format: BibliographyFormat {
                entry_separator: "\n".to_string(),
                field_separators: {
                    let mut separators = HashMap::new();
                    separators.insert("author_title".to_string(), ". ".to_string());
                    separators.insert("title_venue".to_string(), ". ".to_string());
                    separators
                },
                name_format: NameFormat::LastFirstInitial,
                title_format: TitleFormat::SentenceCase,
                date_format: DateFormat::Year,
                punctuation: PunctuationRules {
                    periods_after_abbreviations: true,
                    commas_between_fields: true,
                    parentheses_around_year: true,
                    quote_titles: false,
                    italicize_journals: true,
                },
            },
            formatting_rules: FormattingRules {
                max_authors: Some(7),
                et_altext: "et al.".to_string(),
                et_al_threshold: 8,
                title_case: false,
                abbreviate_journals: false,
                include_doi: true,
                include_url: false,
            },
            sorting_rules: SortingRules {
                primary_sort: SortField::Author,
                secondary_sort: Some(SortField::Year),
                sort_direction: SortDirection::Ascending,
                group_by_type: false,
            },
        }
    }

    fn create_ieee_style() -> CitationStyle {
        CitationStyle {
            name: "IEEE".to_string(),
            description: "Institute of Electrical and Electronics Engineers style".to_string(),
            intext_format: InTextFormat::Numbered,
            bibliography_format: BibliographyFormat {
                entry_separator: "\n".to_string(),
                field_separators: HashMap::new(),
                name_format: NameFormat::FirstInitialLast,
                title_format: TitleFormat::AsEntered,
                date_format: DateFormat::Year,
                punctuation: PunctuationRules {
                    periods_after_abbreviations: true,
                    commas_between_fields: true,
                    parentheses_around_year: false,
                    quote_titles: true,
                    italicize_journals: true,
                },
            },
            formatting_rules: FormattingRules {
                max_authors: None,
                et_altext: "et al.".to_string(),
                et_al_threshold: 7,
                title_case: false,
                abbreviate_journals: true,
                include_doi: true,
                include_url: false,
            },
            sorting_rules: SortingRules {
                primary_sort: SortField::Year,
                secondary_sort: Some(SortField::Author),
                sort_direction: SortDirection::Ascending,
                group_by_type: false,
            },
        }
    }

    fn create_acm_style() -> CitationStyle {
        CitationStyle {
            name: "ACM".to_string(),
            description: "Association for Computing Machinery style".to_string(),
            intext_format: InTextFormat::Numbered,
            bibliography_format: BibliographyFormat {
                entry_separator: "\n".to_string(),
                field_separators: HashMap::new(),
                name_format: NameFormat::FirstMiddleLast,
                title_format: TitleFormat::TitleCase,
                date_format: DateFormat::Year,
                punctuation: PunctuationRules {
                    periods_after_abbreviations: true,
                    commas_between_fields: true,
                    parentheses_around_year: false,
                    quote_titles: false,
                    italicize_journals: true,
                },
            },
            formatting_rules: FormattingRules {
                max_authors: None,
                et_altext: "et al.".to_string(),
                et_al_threshold: 3,
                title_case: true,
                abbreviate_journals: false,
                include_doi: true,
                include_url: true,
            },
            sorting_rules: SortingRules {
                primary_sort: SortField::Author,
                secondary_sort: Some(SortField::Year),
                sort_direction: SortDirection::Ascending,
                group_by_type: false,
            },
        }
    }
}

impl BibTeXProcessor {
    /// Create a new BibTeX processor
    pub fn new(settings: BibTeXSettings) -> Self {
        Self { settings }
    }

    /// Parse BibTeX content into citations
    pub fn parse_bibtex(&self, content: &str) -> Result<Vec<Citation>> {
        // Simplified BibTeX parser
        // In a real implementation, you'd want a proper BibTeX parser
        let mut citations = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut current_entry: Option<(String, PublicationType, HashMap<String, String>)> = None;

        for line in lines {
            let line = line.trim();

            if line.starts_with('@') {
                // Save previous entry
                if let Some((key, pub_type, fields)) = current_entry.take() {
                    if let Ok(citation) = self.fields_to_citation(key, pub_type, fields) {
                        citations.push(citation);
                    }
                }

                // Parse new entry
                if let Some(pos) = line.find('{') {
                    let entry_type = line[1..pos].to_lowercase();
                    let pub_type = self.bibtex_type_to_publication_type(&entry_type);

                    let key_part = &line[pos + 1..];
                    if let Some(comma_pos) = key_part.find(',') {
                        let key = key_part[..comma_pos].trim().to_string();
                        current_entry = Some((key, pub_type, HashMap::new()));
                    }
                }
            } else if line.contains('=') && current_entry.is_some() {
                // Parse field
                if let Some(eq_pos) = line.find('=') {
                    let field_name = line[..eq_pos].trim().to_lowercase();
                    let field_value = line[eq_pos + 1..]
                        .trim()
                        .trim_start_matches('{')
                        .trim_end_matches("},")
                        .trim_start_matches('"')
                        .trim_end_matches("\",")
                        .to_string();

                    if let Some((_, _, ref mut fields)) = current_entry {
                        fields.insert(field_name, field_value);
                    }
                }
            }
        }

        // Save last entry
        if let Some((key, pub_type, fields)) = current_entry {
            if let Ok(citation) = self.fields_to_citation(key, pub_type, fields) {
                citations.push(citation);
            }
        }

        Ok(citations)
    }

    fn bibtex_type_to_publication_type(&self, bibtex_type: &str) -> PublicationType {
        match bibtex_type {
            "article" => PublicationType::Article,
            "inproceedings" | "conference" => PublicationType::InProceedings,
            "book" => PublicationType::Book,
            "incollection" | "inbook" => PublicationType::InCollection,
            "phdthesis" => PublicationType::PhDThesis,
            "mastersthesis" => PublicationType::MastersThesis,
            "techreport" => PublicationType::TechReport,
            "manual" => PublicationType::Manual,
            "unpublished" => PublicationType::Unpublished,
            _ => PublicationType::Misc,
        }
    }

    fn fields_to_citation(
        &self,
        key: String,
        pub_type: PublicationType,
        fields: HashMap<String, String>,
    ) -> Result<Citation> {
        let title = fields.get("title").cloned().unwrap_or_default();

        // Parse authors
        let authors = if let Some(author_str) = fields.get("author") {
            self.parse_authors(author_str)
        } else {
            Vec::new()
        };

        // Parse year
        let year = fields.get("year").and_then(|y| y.parse().ok());

        // Determine venue field based on publication type
        let venue = match pub_type {
            PublicationType::Article => fields.get("journal").cloned(),
            PublicationType::InProceedings => fields.get("booktitle").cloned(),
            PublicationType::Book => fields.get("publisher").cloned(),
            PublicationType::InCollection => fields.get("booktitle").cloned(),
            PublicationType::PhDThesis => fields.get("school").cloned(),
            PublicationType::MastersThesis => fields.get("school").cloned(),
            PublicationType::TechReport => fields.get("institution").cloned(),
            PublicationType::Manual => fields.get("organization").cloned(),
            PublicationType::Misc => fields.get("howpublished").cloned(),
            PublicationType::Unpublished => fields.get("note").cloned(),
            PublicationType::Preprint => fields.get("archivePrefix").cloned(),
            PublicationType::Patent => fields.get("assignee").cloned(),
            PublicationType::Software => fields.get("url").cloned(),
            PublicationType::Dataset => fields.get("url").cloned(),
        };

        let now = Utc::now();

        Ok(Citation {
            key,
            publication_type: pub_type,
            title,
            authors,
            year,
            venue,
            volume: fields.get("volume").cloned(),
            issue: fields.get("number").cloned(),
            pages: fields.get("pages").cloned(),
            doi: fields.get("doi").cloned(),
            url: fields.get("url").cloned(),
            abstracttext: fields.get("abstract").cloned(),
            keywords: Vec::new(),
            notes: fields.get("note").cloned(),
            custom_fields: HashMap::new(),
            attachments: Vec::new(),
            groups: Vec::new(),
            import_source: Some("BibTeX".to_string()),
            created_at: now,
            modified_at: now,
        })
    }

    fn parse_authors(&self, author_str: &str) -> Vec<Author> {
        author_str
            .split(" and ")
            .map(|author_part| {
                let author_part = author_part.trim();
                if let Some(comma_pos) = author_part.find(',') {
                    // "Last, First" format
                    let last_name = author_part[..comma_pos].trim().to_string();
                    let first_name = author_part[comma_pos + 1..].trim().to_string();
                    Author {
                        first_name,
                        last_name,
                        middle_name: None,
                        suffix: None,
                        orcid: None,
                        affiliation: None,
                    }
                } else {
                    // "First Last" format
                    let parts: Vec<&str> = author_part.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let first_name = parts[0].to_string();
                        let last_name = parts[parts.len() - 1].to_string();
                        let middle_name = if parts.len() > 2 {
                            Some(parts[1..parts.len() - 1].join(" "))
                        } else {
                            None
                        };
                        Author {
                            first_name,
                            last_name,
                            middle_name,
                            suffix: None,
                            orcid: None,
                            affiliation: None,
                        }
                    } else {
                        // Single name
                        Author {
                            first_name: String::new(),
                            last_name: author_part.to_string(),
                            middle_name: None,
                            suffix: None,
                            orcid: None,
                            affiliation: None,
                        }
                    }
                }
            })
            .collect()
    }
}

impl Default for CitationSettings {
    fn default() -> Self {
        Self {
            auto_generate_keys: true,
            key_pattern: "{author}{year}".to_string(),
            auto_import_doi: true,
            auto_import_url: false,
            duplicate_detection: true,
            backup_enabled: true,
            export_formats: vec![ExportFormat::BibTeX, ExportFormat::RIS],
        }
    }
}

impl Default for BibTeXSettings {
    fn default() -> Self {
        Self {
            preserve_case: true,
            utf8_conversion: true,
            cleanup_formatting: true,
            validate_entries: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_citation_manager_creation() {
        let manager = CitationManager::new();

        assert!(manager.styles.contains_key("APA"));
        assert!(manager.styles.contains_key("IEEE"));
        assert!(manager.styles.contains_key("ACM"));
        assert_eq!(manager.default_style, "APA");
    }

    #[test]
    fn test_add_citation() {
        let mut manager = CitationManager::new();

        let citation = Citation {
            key: "test2023".to_string(),
            publication_type: PublicationType::Article,
            title: "Test Article".to_string(),
            authors: vec![Author {
                first_name: "John".to_string(),
                last_name: "Doe".to_string(),
                middle_name: None,
                suffix: None,
                orcid: None,
                affiliation: None,
            }],
            year: Some(2023),
            venue: Some("Test Journal".to_string()),
            volume: None,
            issue: None,
            pages: None,
            doi: None,
            url: None,
            abstracttext: None,
            keywords: Vec::new(),
            notes: None,
            custom_fields: HashMap::new(),
            attachments: Vec::new(),
            groups: Vec::new(),
            import_source: None,
            created_at: Utc::now(),
            modified_at: Utc::now(),
        };

        assert!(manager.add_citation(citation).is_ok());
        assert!(manager.citations.contains_key("test2023"));
    }

    #[test]
    fn test_search_citations() {
        let mut manager = CitationManager::new();

        let citation = Citation {
            key: "test2023".to_string(),
            publication_type: PublicationType::Article,
            title: "Machine Learning Optimization".to_string(),
            authors: vec![Author {
                first_name: "Jane".to_string(),
                last_name: "Smith".to_string(),
                middle_name: None,
                suffix: None,
                orcid: None,
                affiliation: None,
            }],
            year: Some(2023),
            venue: None,
            volume: None,
            issue: None,
            pages: None,
            doi: None,
            url: None,
            abstracttext: None,
            keywords: vec!["optimization".to_string(), "machine learning".to_string()],
            notes: None,
            custom_fields: HashMap::new(),
            attachments: Vec::new(),
            groups: Vec::new(),
            import_source: None,
            created_at: Utc::now(),
            modified_at: Utc::now(),
        };

        manager.add_citation(citation).unwrap();

        let results = manager.search_citations("optimization");
        assert_eq!(results.len(), 1);

        let results = manager.search_citations("Smith");
        assert_eq!(results.len(), 1);
    }
}
