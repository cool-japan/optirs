// Publication generation and management for academic research
//
// This module provides tools for generating academic publications from experimental
// results, managing bibliographies, and formatting papers for various venues.

use crate::error::Result;
use crate::research::experiments::{Experiment, ExperimentResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Academic publication representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Publication {
    /// Publication identifier
    pub id: String,
    /// Publication title
    pub title: String,
    /// Publication abstract
    pub abstracttext: String,
    /// Authors
    pub authors: Vec<Author>,
    /// Publication type
    pub publication_type: PublicationType,
    /// Venue information
    pub venue: Option<Venue>,
    /// Publication status
    pub status: PublicationStatus,
    /// Keywords
    pub keywords: Vec<String>,
    /// Manuscript sections
    pub sections: Vec<ManuscriptSection>,
    /// Bibliography
    pub bibliography: Bibliography,
    /// Associated experiments
    pub experiment_ids: Vec<String>,
    /// Submission history
    pub submission_history: Vec<SubmissionRecord>,
    /// Review information
    pub reviews: Vec<Review>,
    /// Publication metadata
    pub metadata: PublicationMetadata,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
}

/// Author information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Author {
    /// Full name
    pub name: String,
    /// Email address
    pub email: String,
    /// Affiliations
    pub affiliations: Vec<Affiliation>,
    /// ORCID identifier
    pub orcid: Option<String>,
    /// Author position (first, corresponding, etc.)
    pub position: AuthorPosition,
    /// Contribution description
    pub contributions: Vec<String>,
}

/// Author affiliation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Affiliation {
    /// Institution name
    pub institution: String,
    /// Department
    pub department: Option<String>,
    /// Address
    pub address: String,
    /// Country
    pub country: String,
}

/// Author position/role
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuthorPosition {
    /// First author
    First,
    /// Corresponding author
    Corresponding,
    /// Senior author
    Senior,
    /// Equal contribution
    EqualContribution,
    /// Regular author
    Regular,
}

/// Publication types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PublicationType {
    /// Conference paper
    ConferencePaper,
    /// Journal article
    JournalArticle,
    /// Workshop paper
    WorkshopPaper,
    /// Technical report
    TechnicalReport,
    /// Preprint
    Preprint,
    /// Thesis
    Thesis,
    /// Book chapter
    BookChapter,
    /// Patent
    Patent,
    /// Software paper
    SoftwarePaper,
    /// Dataset paper
    DatasetPaper,
}

/// Publication venue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Venue {
    /// Venue name
    pub name: String,
    /// Venue type
    pub venue_type: VenueType,
    /// Abbreviation
    pub abbreviation: Option<String>,
    /// Publisher
    pub publisher: Option<String>,
    /// Impact factor
    pub impact_factor: Option<f64>,
    /// H-index
    pub h_index: Option<u32>,
    /// Acceptance rate
    pub acceptance_rate: Option<f64>,
    /// Ranking (A*, A, B, C)
    pub ranking: Option<String>,
    /// Venue URL
    pub url: Option<String>,
}

/// Venue types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VenueType {
    /// Academic conference
    Conference,
    /// Academic journal
    Journal,
    /// Workshop
    Workshop,
    /// Symposium
    Symposium,
    /// Preprint server
    PreprintServer,
    /// Repository
    Repository,
}

/// Publication status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PublicationStatus {
    /// Draft in preparation
    Draft,
    /// Ready for submission
    ReadyForSubmission,
    /// Submitted
    Submitted,
    /// Under review
    UnderReview,
    /// Revision requested
    RevisionRequested,
    /// Accepted
    Accepted,
    /// Published
    Published,
    /// Rejected
    Rejected,
    /// Withdrawn
    Withdrawn,
}

/// Manuscript section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManuscriptSection {
    /// Section title
    pub title: String,
    /// Section content
    pub content: String,
    /// Section order
    pub order: usize,
    /// Section type
    pub section_type: SectionType,
    /// Word count
    pub word_count: usize,
    /// Figures and tables
    pub figures: Vec<Figure>,
    pub tables: Vec<Table>,
    /// References in this section
    pub references: Vec<String>,
}

/// Section types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SectionType {
    /// Abstract
    Abstract,
    /// Introduction
    Introduction,
    /// Background/Related Work
    RelatedWork,
    /// Methodology
    Methodology,
    /// Experiments
    Experiments,
    /// Results
    Results,
    /// Discussion
    Discussion,
    /// Conclusion
    Conclusion,
    /// Acknowledgments
    Acknowledgments,
    /// References
    References,
    /// Appendix
    Appendix,
    /// Custom section
    Custom(String),
}

/// Figure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Figure {
    /// Figure caption
    pub caption: String,
    /// Figure file path
    pub file_path: PathBuf,
    /// Figure type
    pub figure_type: FigureType,
    /// Figure number
    pub number: usize,
    /// Width (in publication units)
    pub width: Option<f64>,
    /// Height (in publication units)
    pub height: Option<f64>,
    /// Associated experiment ID
    pub experiment_id: Option<String>,
}

/// Figure types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FigureType {
    /// Plot/graph
    Plot,
    /// Diagram
    Diagram,
    /// Algorithm flowchart
    Flowchart,
    /// Architecture diagram
    Architecture,
    /// Screenshot
    Screenshot,
    /// Photo
    Photo,
    /// Other
    Other,
}

/// Table information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    /// Table caption
    pub caption: String,
    /// Table data
    pub data: Vec<Vec<String>>,
    /// Column headers
    pub headers: Vec<String>,
    /// Table number
    pub number: usize,
    /// Associated experiment ID
    pub experiment_id: Option<String>,
}

/// Bibliography management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bibliography {
    /// BibTeX entries
    pub entries: HashMap<String, BibTeXEntry>,
    /// Citation style
    pub citation_style: CitationStyle,
    /// Bibliography file path
    pub file_path: Option<PathBuf>,
}

/// BibTeX entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BibTeXEntry {
    /// Entry key
    pub key: String,
    /// Entry type (article, inproceedings, etc.)
    pub entry_type: String,
    /// Fields (title, author, year, etc.)
    pub fields: HashMap<String, String>,
}

/// Citation styles
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CitationStyle {
    /// APA style
    APA,
    /// IEEE style
    IEEE,
    /// ACM style
    ACM,
    /// Nature style
    Nature,
    /// Science style
    Science,
    /// Chicago style
    Chicago,
    /// MLA style
    MLA,
    /// Harvard style
    Harvard,
    /// Custom style
    Custom(String),
}

/// Submission record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmissionRecord {
    /// Submission timestamp
    pub submitted_at: DateTime<Utc>,
    /// Venue submitted to
    pub venue: Venue,
    /// Submission ID
    pub submission_id: Option<String>,
    /// Submission status
    pub status: SubmissionStatus,
    /// Decision date
    pub decision_date: Option<DateTime<Utc>>,
    /// Decision outcome
    pub decision: Option<Decision>,
    /// Comments from editors
    pub editor_comments: Option<String>,
}

/// Submission status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SubmissionStatus {
    /// Submitted
    Submitted,
    /// Under review
    UnderReview,
    /// Decision made
    Decided,
    /// Withdrawn
    Withdrawn,
}

/// Review decision
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Decision {
    /// Accept
    Accept,
    /// Accept with minor revisions
    AcceptMinorRevisions,
    /// Accept with major revisions
    AcceptMajorRevisions,
    /// Reject and resubmit
    RejectAndResubmit,
    /// Reject
    Reject,
}

/// Review information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Review {
    /// Review ID
    pub id: String,
    /// Reviewer information (anonymous)
    pub reviewer: ReviewerInfo,
    /// Overall score
    pub overall_score: Option<f64>,
    /// Confidence score
    pub confidence_score: Option<f64>,
    /// Detailed scores
    pub detailed_scores: HashMap<String, f64>,
    /// Written review
    pub reviewtext: String,
    /// Strengths
    pub strengths: Vec<String>,
    /// Weaknesses
    pub weaknesses: Vec<String>,
    /// Questions for authors
    pub questions: Vec<String>,
    /// Recommendation
    pub recommendation: ReviewRecommendation,
    /// Review timestamp
    pub submitted_at: DateTime<Utc>,
}

/// Reviewer information (anonymized)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewerInfo {
    /// Anonymous reviewer ID
    pub anonymous_id: String,
    /// Expertise level
    pub expertise_level: ExpertiseLevel,
    /// Research areas
    pub research_areas: Vec<String>,
}

/// Expertise levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExpertiseLevel {
    /// Expert in the field
    Expert,
    /// Knowledgeable
    Knowledgeable,
    /// Some knowledge
    SomeKnowledge,
    /// Limited knowledge
    LimitedKnowledge,
}

/// Review recommendation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReviewRecommendation {
    /// Strong accept
    StrongAccept,
    /// Accept
    Accept,
    /// Weak accept
    WeakAccept,
    /// Borderline
    Borderline,
    /// Weak reject
    WeakReject,
    /// Reject
    Reject,
    /// Strong reject
    StrongReject,
}

/// Publication metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PublicationMetadata {
    /// DOI
    pub doi: Option<String>,
    /// ArXiv ID
    pub arxiv_id: Option<String>,
    /// Page numbers
    pub pages: Option<String>,
    /// Volume
    pub volume: Option<String>,
    /// Issue/Number
    pub issue: Option<String>,
    /// Publication year
    pub year: Option<u32>,
    /// Publication month
    pub month: Option<u32>,
    /// ISBN/ISSN
    pub isbn_issn: Option<String>,
    /// License
    pub license: Option<String>,
    /// Open access status
    pub open_access: bool,
}

/// Publication generator for creating publications from experiments
#[derive(Debug)]
pub struct PublicationGenerator {
    /// Template repository
    templates: HashMap<PublicationType, PublicationTemplate>,
    /// Default citation style
    default_citation_style: CitationStyle,
    /// Output directory
    output_dir: PathBuf,
}

/// Publication template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicationTemplate {
    /// Template name
    pub name: String,
    /// Template sections
    pub sections: Vec<SectionTemplate>,
    /// Default formatting options
    pub formatting: FormattingOptions,
    /// Target venue constraints
    pub venue_constraints: VenueConstraints,
}

/// Section template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionTemplate {
    /// Section type
    pub section_type: SectionType,
    /// Template content
    pub template: String,
    /// Required fields
    pub required_fields: Vec<String>,
    /// Word count target
    pub target_word_count: Option<usize>,
}

/// Formatting options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattingOptions {
    /// Document format
    pub format: DocumentFormat,
    /// Font size
    pub font_size: u32,
    /// Line spacing
    pub line_spacing: f64,
    /// Margins (in cm)
    pub margins: Margins,
    /// Citation format
    pub citation_format: CitationFormat,
    /// Figure numbering
    pub figure_numbering: NumberingStyle,
    /// Table numbering
    pub table_numbering: NumberingStyle,
}

/// Document formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DocumentFormat {
    /// LaTeX
    LaTeX,
    /// Markdown
    Markdown,
    /// HTML
    HTML,
    /// Microsoft Word
    Word,
    /// PDF
    PDF,
}

/// Page margins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Margins {
    /// Top margin
    pub top: f64,
    /// Bottom margin
    pub bottom: f64,
    /// Left margin
    pub left: f64,
    /// Right margin
    pub right: f64,
}

/// Citation format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CitationFormat {
    /// Numbered citations \[1\]
    Numbered,
    /// Author-year citations (Author, 2023)
    AuthorYear,
    /// Superscript citations¹
    Superscript,
    /// Footnote citations
    Footnote,
}

/// Numbering styles
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NumberingStyle {
    /// Arabic numerals (1, 2, 3)
    Arabic,
    /// Roman numerals (I, II, III)
    Roman,
    /// Letters (a, b, c)
    Letters,
    /// No numbering
    None,
}

/// Venue constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueConstraints {
    /// Maximum word count
    pub max_word_count: Option<usize>,
    /// Maximum page count
    pub max_page_count: Option<usize>,
    /// Required sections
    pub required_sections: Vec<SectionType>,
    /// Forbidden sections
    pub forbidden_sections: Vec<SectionType>,
    /// Figure limits
    pub max_figures: Option<usize>,
    /// Table limits
    pub max_tables: Option<usize>,
    /// Reference limits
    pub max_references: Option<usize>,
}

impl Publication {
    /// Create a new publication
    pub fn new(title: &str) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            title: title.to_string(),
            abstracttext: String::new(),
            authors: Vec::new(),
            publication_type: PublicationType::ConferencePaper,
            venue: None,
            status: PublicationStatus::Draft,
            keywords: Vec::new(),
            sections: Vec::new(),
            bibliography: Bibliography::new(),
            experiment_ids: Vec::new(),
            submission_history: Vec::new(),
            reviews: Vec::new(),
            metadata: PublicationMetadata::default(),
            created_at: now,
            modified_at: now,
        }
    }

    /// Set publication abstract
    pub fn abstracttext(mut self, abstracttext: &str) -> Self {
        self.abstracttext = abstracttext.to_string();
        self.modified_at = Utc::now();
        self
    }

    /// Add an author
    pub fn add_author(mut self, author: Author) -> Self {
        self.authors.push(author);
        self.modified_at = Utc::now();
        self
    }

    /// Set publication type
    pub fn publication_type(mut self, pubtype: PublicationType) -> Self {
        self.publication_type = pubtype;
        self.modified_at = Utc::now();
        self
    }

    /// Set target venue
    pub fn venue(mut self, venue: Venue) -> Self {
        self.venue = Some(venue);
        self.modified_at = Utc::now();
        self
    }

    /// Add keywords
    pub fn keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = keywords;
        self.modified_at = Utc::now();
        self
    }

    /// Associate with experiment
    pub fn add_experiment(&mut self, experiment_id: &str) {
        self.experiment_ids.push(experiment_id.to_string());
        self.modified_at = Utc::now();
    }

    /// Add a manuscript section
    pub fn add_section(&mut self, section: ManuscriptSection) {
        self.sections.push(section);
        self.modified_at = Utc::now();
    }

    /// Generate LaTeX document
    pub fn generate_latex(&self) -> Result<String> {
        let mut latex = String::new();

        // Document class and packages
        latex.push_str("\\documentclass[conference]{IEEEtran}\n");
        latex.push_str("\\usepackage{graphicx}\n");
        latex.push_str("\\usepackage{booktabs}\n");
        latex.push_str("\\usepackage{amsmath}\n");
        latex.push_str("\\usepackage{url}\n\n");

        latex.push_str("\\begin{document}\n\n");

        // Title and authors
        latex.push_str(&format!("\\title{{{}}}\n\n", self.title));

        latex.push_str("\\author{\n");
        for (i, author) in self.authors.iter().enumerate() {
            if i > 0 {
                latex.push_str("\\and\n");
            }
            latex.push_str(&format!("\\IEEEauthorblockN{{{}}}\n", author.name));
            if !author.affiliations.is_empty() {
                latex.push_str(&format!(
                    "\\IEEEauthorblockA{{{}}}\n",
                    author.affiliations[0].institution
                ));
            }
        }
        latex.push_str("}\n\n");

        latex.push_str("\\maketitle\n\n");

        // Abstract
        if !self.abstracttext.is_empty() {
            latex.push_str("\\begin{abstract}\n");
            latex.push_str(&self.abstracttext);
            latex.push_str("\n\\end{abstract}\n\n");
        }

        // Keywords
        if !self.keywords.is_empty() {
            latex.push_str("\\begin{IEEEkeywords}\n");
            latex.push_str(&self.keywords.join(", "));
            latex.push_str("\n\\end{IEEEkeywords}\n\n");
        }

        // Sections
        let mut sorted_sections = self.sections.clone();
        sorted_sections.sort_by_key(|s| s.order);

        for section in sorted_sections {
            match section.section_type {
                SectionType::Abstract => continue,   // Already handled
                SectionType::References => continue, // Handle at end
                _ => {
                    latex.push_str(&format!("\\section{{{}}}\n", section.title));
                    latex.push_str(&section.content);
                    latex.push_str("\n\n");
                }
            }
        }

        // Bibliography
        if !self.bibliography.entries.is_empty() {
            latex.push_str("\\begin{thebibliography}{99}\n");
            for entry in self.bibliography.entries.values() {
                latex.push_str(&self.format_bibtex_entry_latex(entry));
            }
            latex.push_str("\\end{thebibliography}\n\n");
        }

        latex.push_str("\\end{document}\n");

        Ok(latex)
    }

    fn format_bibtex_entry_latex(&self, entry: &BibTeXEntry) -> String {
        format!(
            "\\bibitem{{{}}}\n{}\n\n",
            entry.key,
            self.format_bibtex_fields(&entry.fields)
        )
    }

    fn format_bibtex_fields(&self, fields: &HashMap<String, String>) -> String {
        let mut result = String::new();

        if let Some(author) = fields.get("author") {
            result.push_str(author);
        }

        if let Some(title) = fields.get("title") {
            result.push_str(&format!(", ``{}'', ", title));
        }

        if let Some(journal) = fields.get("journal") {
            result.push_str(&format!("\\emph{{{}}}, ", journal));
        } else if let Some(booktitle) = fields.get("booktitle") {
            result.push_str(&format!("in \\emph{{{}}}, ", booktitle));
        }

        if let Some(year) = fields.get("year") {
            result.push_str(year);
        }

        result
    }

    /// Generate markdown document
    pub fn generate_markdown(&self) -> Result<String> {
        let mut markdown = String::new();

        // Title
        markdown.push_str(&format!("# {}\n\n", self.title));

        // Authors
        if !self.authors.is_empty() {
            markdown.push_str("**Authors**: ");
            let author_names: Vec<String> = self.authors.iter().map(|a| a.name.clone()).collect();
            markdown.push_str(&author_names.join(", "));
            markdown.push_str("\n\n");
        }

        // Abstract
        if !self.abstracttext.is_empty() {
            markdown.push_str("## Abstract\n\n");
            markdown.push_str(&self.abstracttext);
            markdown.push_str("\n\n");
        }

        // Keywords
        if !self.keywords.is_empty() {
            markdown.push_str("**Keywords**: ");
            markdown.push_str(&self.keywords.join(", "));
            markdown.push_str("\n\n");
        }

        // Sections
        let mut sorted_sections = self.sections.clone();
        sorted_sections.sort_by_key(|s| s.order);

        for section in sorted_sections {
            match section.section_type {
                SectionType::Abstract => continue, // Already handled
                _ => {
                    markdown.push_str(&format!("## {}\n\n", section.title));
                    markdown.push_str(&section.content);
                    markdown.push_str("\n\n");
                }
            }
        }

        // References
        if !self.bibliography.entries.is_empty() {
            markdown.push_str("## References\n\n");
            for (i, entry) in self.bibliography.entries.values().enumerate() {
                markdown.push_str(&format!(
                    "{}. {}\n",
                    i + 1,
                    self.format_bibtex_entry_markdown(entry)
                ));
            }
        }

        Ok(markdown)
    }

    fn format_bibtex_entry_markdown(&self, entry: &BibTeXEntry) -> String {
        let mut result = String::new();

        if let Some(author) = entry.fields.get("author") {
            result.push_str(author);
        }

        if let Some(title) = entry.fields.get("title") {
            result.push_str(&format!(". \"{}\". ", title));
        }

        if let Some(journal) = entry.fields.get("journal") {
            result.push_str(&format!("*{}*. ", journal));
        } else if let Some(booktitle) = entry.fields.get("booktitle") {
            result.push_str(&format!("In *{}*. ", booktitle));
        }

        if let Some(year) = entry.fields.get("year") {
            result.push_str(year);
        }

        result
    }

    /// Generate submission statistics
    pub fn submission_statistics(&self) -> SubmissionStatistics {
        let total_submissions = self.submission_history.len();
        let accepted = self
            .submission_history
            .iter()
            .filter(|s| {
                matches!(
                    s.decision,
                    Some(Decision::Accept)
                        | Some(Decision::AcceptMinorRevisions)
                        | Some(Decision::AcceptMajorRevisions)
                )
            })
            .count();
        let rejected = self
            .submission_history
            .iter()
            .filter(|s| matches!(s.decision, Some(Decision::Reject)))
            .count();

        let avg_review_time = if !self.submission_history.is_empty() {
            let total_days: i64 = self
                .submission_history
                .iter()
                .filter_map(|s| {
                    s.decision_date
                        .map(|decision_date| (decision_date - s.submitted_at).num_days())
                })
                .sum();
            total_days as f64 / self.submission_history.len() as f64
        } else {
            0.0
        };

        SubmissionStatistics {
            total_submissions,
            accepted,
            rejected,
            pending: total_submissions - accepted - rejected,
            acceptance_rate: if total_submissions > 0 {
                accepted as f64 / total_submissions as f64
            } else {
                0.0
            },
            avg_review_time_days: avg_review_time,
        }
    }
}

/// Submission statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmissionStatistics {
    /// Total number of submissions
    pub total_submissions: usize,
    /// Number of accepted submissions
    pub accepted: usize,
    /// Number of rejected submissions
    pub rejected: usize,
    /// Number of pending submissions
    pub pending: usize,
    /// Acceptance rate (0.0 to 1.0)
    pub acceptance_rate: f64,
    /// Average review time in days
    pub avg_review_time_days: f64,
}

impl Default for Bibliography {
    fn default() -> Self {
        Self::new()
    }
}

impl Bibliography {
    /// Create a new bibliography
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            citation_style: CitationStyle::IEEE,
            file_path: None,
        }
    }

    /// Add a BibTeX entry
    pub fn add_entry(&mut self, entry: BibTeXEntry) {
        self.entries.insert(entry.key.clone(), entry);
    }

    /// Load from BibTeX file
    pub fn load_bibtex_file(&mut self, filepath: &PathBuf) -> Result<()> {
        let content = std::fs::read_to_string(filepath)?;
        self.parse_bibtex(&content)?;
        self.file_path = Some(filepath.clone());
        Ok(())
    }

    /// Parse BibTeX content
    pub fn parse_bibtex(&mut self, content: &str) -> Result<()> {
        // Simplified BibTeX parser
        // In a real implementation, you'd want a proper BibTeX parser
        let lines: Vec<&str> = content.lines().collect();
        let mut current_entry: Option<BibTeXEntry> = None;

        for line in lines {
            let line = line.trim();
            if line.starts_with('@') {
                // Save previous entry
                if let Some(entry) = current_entry.take() {
                    self.entries.insert(entry.key.clone(), entry);
                }

                // Start new entry
                if let Some(pos) = line.find('{') {
                    let entry_type = line[1..pos].to_lowercase();
                    let key_part = &line[pos + 1..];
                    if let Some(comma_pos) = key_part.find(',') {
                        let key = key_part[..comma_pos].trim().to_string();
                        current_entry = Some(BibTeXEntry {
                            key,
                            entry_type,
                            fields: HashMap::new(),
                        });
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

                    if let Some(ref mut entry) = current_entry {
                        entry.fields.insert(field_name, field_value);
                    }
                }
            }
        }

        // Save last entry
        if let Some(entry) = current_entry {
            self.entries.insert(entry.key.clone(), entry);
        }

        Ok(())
    }
}

impl PublicationGenerator {
    /// Create a new publication generator
    pub fn new(_outputdir: PathBuf) -> Self {
        Self {
            templates: HashMap::new(),
            default_citation_style: CitationStyle::IEEE,
            output_dir: _outputdir,
        }
    }

    /// Generate publication from experiments
    pub fn generate_from_experiments(
        &self,
        experiments: &[Experiment],
        template: &PublicationTemplate,
    ) -> Result<Publication> {
        let mut publication = Publication::new("Generated Publication");

        // Generate abstract from experiments
        let abstracttext = self.generate_abstract(experiments)?;
        publication.abstracttext = abstracttext;

        // Generate sections
        for section_template in &template.sections {
            let section = self.generate_section(section_template, experiments)?;
            publication.add_section(section);
        }

        Ok(publication)
    }

    fn generate_abstract(&self, experiments: &[Experiment]) -> Result<String> {
        // Generate abstract based on experiments
        let mut abstracttext = String::new();

        abstracttext.push_str(
            "This paper presents experimental results comparing various optimization algorithms. ",
        );

        if !experiments.is_empty() {
            abstracttext.push_str(&format!(
                "We conducted {} experiments evaluating the performance of different optimizers. ",
                experiments.len()
            ));
        }

        abstracttext.push_str("Our results demonstrate significant differences in convergence behavior and final performance across different optimization methods.");

        Ok(abstracttext)
    }

    fn generate_section(
        &self,
        template: &SectionTemplate,
        experiments: &[Experiment],
    ) -> Result<ManuscriptSection> {
        let content = match template.section_type {
            SectionType::Introduction => self.generate_introduction(experiments)?,
            SectionType::Methodology => self.generate_methodology(experiments)?,
            SectionType::Experiments => self.generate_experiments_section(experiments)?,
            SectionType::Results => self.generate_results(experiments)?,
            SectionType::Conclusion => self.generate_conclusion(experiments)?,
            _ => template.template.clone(),
        };

        Ok(ManuscriptSection {
            title: match template.section_type {
                SectionType::Abstract => "Abstract".to_string(),
                SectionType::Introduction => "Introduction".to_string(),
                SectionType::RelatedWork => "Related Work".to_string(),
                SectionType::Methodology => "Methodology".to_string(),
                SectionType::Experiments => "Experiments".to_string(),
                SectionType::Results => "Results".to_string(),
                SectionType::Discussion => "Discussion".to_string(),
                SectionType::Conclusion => "Conclusion".to_string(),
                SectionType::Acknowledgments => "Acknowledgments".to_string(),
                SectionType::References => "References".to_string(),
                SectionType::Appendix => "Appendix".to_string(),
                SectionType::Custom(ref name) => name.clone(),
            },
            content,
            order: 0, // Will be set based on section type
            section_type: template.section_type.clone(),
            word_count: 0, // Will be calculated
            figures: Vec::new(),
            tables: Vec::new(),
            references: Vec::new(),
        })
    }

    fn generate_introduction(&self, experiments: &[Experiment]) -> Result<String> {
        Ok("This section introduces the research problem and motivation for comparing optimization algorithms.".to_string())
    }

    fn generate_methodology(&self, experiments: &[Experiment]) -> Result<String> {
        let mut content = String::new();
        content.push_str("We evaluate the following optimization algorithms:\n\n");

        for experiment in experiments {
            for optimizer_name in experiment.optimizer_configs.keys() {
                content.push_str(&format!("- {}\n", optimizer_name));
            }
        }

        Ok(content)
    }

    fn generate_experiments_section(&self, experiments: &[Experiment]) -> Result<String> {
        let mut content = String::new();
        content.push_str("We conducted the following experiments:\n\n");

        for experiment in experiments {
            content.push_str(&format!(
                "**{}**: {}\n\n",
                experiment.name, experiment.description
            ));
        }

        Ok(content)
    }

    fn generate_results(&self, experiments: &[Experiment]) -> Result<String> {
        let mut content = String::new();
        content.push_str("The experimental results are summarized below:\n\n");

        for experiment in experiments {
            if !experiment.results.is_empty() {
                content.push_str(&format!("### {}\n\n", experiment.name));
                content.push_str(&format!("Number of runs: {}\n\n", experiment.results.len()));
            }
        }

        Ok(content)
    }

    fn generate_conclusion(&self, experiments: &[Experiment]) -> Result<String> {
        Ok("This section summarizes the key findings and implications of the experimental results.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_publication_creation() {
        let publication = Publication::new("Test Publication")
            .abstracttext("Test abstract")
            .publication_type(PublicationType::ConferencePaper)
            .keywords(vec![
                "optimization".to_string(),
                "machine learning".to_string(),
            ]);

        assert_eq!(publication.title, "Test Publication");
        assert_eq!(publication.abstracttext, "Test abstract");
        assert_eq!(
            publication.publication_type,
            PublicationType::ConferencePaper
        );
        assert_eq!(publication.keywords.len(), 2);
    }

    #[test]
    fn test_bibliography() {
        let mut bibliography = Bibliography::new();

        let entry = BibTeXEntry {
            key: "test2023".to_string(),
            entry_type: "article".to_string(),
            fields: {
                let mut fields = HashMap::new();
                fields.insert("author".to_string(), "Test Author".to_string());
                fields.insert("title".to_string(), "Test Title".to_string());
                fields.insert("year".to_string(), "2023".to_string());
                fields
            },
        };

        bibliography.add_entry(entry);
        assert_eq!(bibliography.entries.len(), 1);
        assert!(bibliography.entries.contains_key("test2023"));
    }

    #[test]
    fn test_markdown_generation() {
        let mut publication = Publication::new("Test Paper");
        publication.abstracttext = "This is a test abstract.".to_string();
        publication.keywords = vec!["test".to_string(), "paper".to_string()];

        let markdown = publication.generate_markdown().unwrap();
        assert!(markdown.contains("# Test Paper"));
        assert!(markdown.contains("## Abstract"));
        assert!(markdown.contains("This is a test abstract."));
        assert!(markdown.contains("**Keywords**: test, paper"));
    }
}
