//! Real, dependency-free parsers for the text/XML reports emitted by
//! external memory-analysis tools (Valgrind's Memcheck, Massif, and
//! HeapTrack).
//!
//! # Why this module exists
//!
//! `bin/memory_leak_reporter.rs` used to read these files only to check
//! whether they existed, then return the exact same hand-typed numbers
//! (`peak_memory: 50 * 1024 * 1024`, `leaked_allocations: vec![... 1024
//! bytes ...]`, etc.) regardless of what the file actually said (F54). A
//! Valgrind report showing zero leaks and one showing a gigabyte leak
//! produced an identical "report". This module actually reads the content.
//!
//! # Honesty rules followed throughout
//!
//! - A file that does not carry any recognizable marker for the tool it
//!   claims to be a report from is a parse **error**, not a silently-empty
//!   result (calling this on the wrong file should not look like "clean
//!   run").
//! - A file that IS recognizable but genuinely reports zero leaks/zero
//!   allocations parses to real zeros -- an honest "nothing found", not an
//!   error.
//! - No field is ever backfilled with an invented value when the tool's
//!   output does not expose it. Where a downstream consumer needs a
//!   placeholder (e.g. `allocation_time_ms`), `0` means "not measured",
//!   matching the zero-means-unmeasured convention already established by
//!   `regression_tester::ResourceMeasurements`.

use crate::error::{OptimError, Result};

/// Parse a human-readable byte size such as `"48.00M"`, `"4.00K"`, `"1.20G"`,
/// `"512B"`, `"1048576"`, or `"48.00MB"`/`"48.00MiB"` into a byte count.
/// Case-insensitive on the suffix. Returns `None` (never a fabricated
/// number) when `text` does not start with a numeric value.
pub fn parse_byte_size(text: &str) -> Option<usize> {
    let text = text.trim();
    let numeric_len = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(text.len());
    if numeric_len == 0 {
        return None;
    }
    let value: f64 = text[..numeric_len].parse().ok()?;
    let suffix = text[numeric_len..].trim().to_ascii_uppercase();
    let multiplier = match suffix.as_str() {
        "" | "B" => 1.0,
        "K" | "KB" | "KIB" => 1024.0,
        "M" | "MB" | "MIB" => 1024.0 * 1024.0,
        "G" | "GB" | "GIB" => 1024.0 * 1024.0 * 1024.0,
        "T" | "TB" | "TIB" => 1024.0_f64.powi(4),
        _ => return None,
    };
    Some((value * multiplier).round() as usize)
}

/// Undo the five predefined XML entities. Order matters: `&amp;` is unescaped
/// last so an input like `&amp;lt;` (a literal ampersand followed by `lt;`)
/// correctly becomes `&lt;`, not `<`.
fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Text content of a `<tag>...</tag>` element occurring (whole) on `line`.
/// Valgrind's `--xml=yes` output puts exactly one element per line for every
/// field this parser reads, so a line-oriented scan (no XML parser
/// dependency) is sufficient and honest -- it never guesses across line
/// boundaries.
fn extract_tag_text(line: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = line.find(&open)? + open.len();
    let end_rel = line[start..].find(&close)?;
    Some(xml_unescape(line[start..start + end_rel].trim()))
}

/// One `<error>` element from Valgrind's Memcheck XML output whose `<kind>`
/// is one of the `Leak_*` variants (a leak report, as opposed to Memcheck's
/// other error kinds like `UninitCondition`, which this parser does not
/// treat as a leak).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValgrindLeakRecord {
    /// From `<xwhat><leakedbytes>`. `0` if the element was absent.
    pub bytes_leaked: usize,
    /// From `<xwhat><leakedblocks>`. `0` if the element was absent.
    pub blocks_leaked: usize,
    /// `<fn>` text from every `<frame>` in the `<stack>`, outermost first.
    pub call_stack: Vec<String>,
    /// Human-readable leak kind: `"definitely lost"`, `"indirectly lost"`,
    /// `"possibly lost"`, or `"still reachable"`.
    pub leak_kind: String,
}

/// A full Memcheck XML report's leak findings.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ValgrindReport {
    pub total_leaks: usize,
    /// Sum of every record's `bytes_leaked` -- a real total, not a constant.
    pub leaked_bytes: usize,
    pub leak_records: Vec<ValgrindLeakRecord>,
}

fn leak_kind_label(raw: &str) -> String {
    match raw {
        "Leak_DefinitelyLost" => "definitely lost",
        "Leak_IndirectlyLost" => "indirectly lost",
        "Leak_PossiblyLost" => "possibly lost",
        "Leak_StillReachable" => "still reachable",
        other => return other.to_string(),
    }
    .to_string()
}

/// Parse Valgrind Memcheck's `--xml=yes` output (typically named
/// `valgrind_memcheck.xml` by CI harnesses).
///
/// Errors when `content` carries neither a `<valgrindoutput` root nor any
/// `<error` element -- i.e. it is not recognizable as this tool's output at
/// all. A recognizable report with zero `Leak_*` errors parses successfully
/// to an all-zero, empty-`leak_records` `ValgrindReport` (a real "no leaks",
/// not a parse failure).
pub fn parse_valgrind_xml(content: &str) -> Result<ValgrindReport> {
    if !content.contains("<valgrindoutput") && !content.contains("<error") {
        return Err(OptimError::InvalidConfig(
            "content does not look like Valgrind XML output (run with --xml=yes): found \
             neither a <valgrindoutput> root nor any <error> element"
                .to_string(),
        ));
    }

    let mut records = Vec::new();
    let mut in_error = false;
    let mut kind: Option<String> = None;
    let mut bytes_leaked = 0usize;
    let mut blocks_leaked = 0usize;
    let mut stack: Vec<String> = Vec::new();

    for raw_line in content.lines() {
        let line = raw_line.trim();

        if line.starts_with("<error>") || line.starts_with("<error ") {
            in_error = true;
            kind = None;
            bytes_leaked = 0;
            blocks_leaked = 0;
            stack.clear();
            continue;
        }
        if !in_error {
            continue;
        }
        if line.starts_with("</error>") {
            if let Some(k) = kind.take() {
                if let Some(stripped) = k.strip_prefix("Leak_") {
                    records.push(ValgrindLeakRecord {
                        bytes_leaked,
                        blocks_leaked,
                        call_stack: std::mem::take(&mut stack),
                        leak_kind: leak_kind_label(&format!("Leak_{stripped}")),
                    });
                }
            }
            in_error = false;
            continue;
        }
        if let Some(k) = extract_tag_text(line, "kind") {
            kind = Some(k);
        } else if let Some(b) = extract_tag_text(line, "leakedbytes") {
            bytes_leaked = b.parse().unwrap_or(0);
        } else if let Some(b) = extract_tag_text(line, "leakedblocks") {
            blocks_leaked = b.parse().unwrap_or(0);
        } else if let Some(f) = extract_tag_text(line, "fn") {
            stack.push(f);
        }
    }

    let leaked_bytes = records.iter().map(|r| r.bytes_leaked).sum();
    Ok(ValgrindReport {
        total_leaks: records.len(),
        leaked_bytes,
        leak_records: records,
    })
}

/// One flattened entry from a Massif `heap_tree=detailed`/`heap_tree=peak`
/// block: `n<child-count>: <bytes> <description>`. `children` is always
/// empty -- see [`MassifReport`] for why nesting is not reconstructed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MassifAllocationNode {
    pub bytes: usize,
    pub function: String,
    pub children: Vec<MassifAllocationNode>,
}

/// A parsed Massif raw report (`ms_print`'s input, i.e. `massif.out.<pid>`,
/// or a copy of it saved as e.g. `massifreport.txt`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MassifReport {
    /// Real maximum of `mem_heap_B + mem_heap_extra_B` across every
    /// snapshot, not a hardcoded 50 MiB.
    pub peak_memory_bytes: usize,
    /// `(time, mem_heap_B + mem_heap_extra_B)` per snapshot, in file order.
    pub memory_timeline: Vec<(u64, usize)>,
    /// Flat list of `n<D>: <bytes> <description>` tree lines found anywhere
    /// in the file. Massif's detailed heap trees nest arbitrarily deep by
    /// indentation; reconstructing that hierarchy has no consumer anywhere
    /// in this crate (`AllocationNode`/`MassifAllocationNode::children` was
    /// dead code even in the pre-fix version), so this parser honestly
    /// extracts every real node it can find without attempting to place it
    /// in the tree -- `children` is therefore always empty, rather than a
    /// fabricated two-node tree.
    pub allocation_tree: Vec<MassifAllocationNode>,
}

fn massif_tree_nodes(content: &str) -> Vec<MassifAllocationNode> {
    let mut nodes = Vec::new();
    for raw_line in content.lines() {
        let line = raw_line.trim_start();
        let Some(rest) = line.strip_prefix('n') else {
            continue;
        };
        let Some(colon) = rest.find(':') else {
            continue;
        };
        let (count_part, after_colon) = rest.split_at(colon);
        if count_part.parse::<u32>().is_err() {
            continue; // Not actually an "n<digits>:" tree line.
        }
        let after_colon = after_colon[1..].trim_start();
        let Some(space) = after_colon.find(' ') else {
            continue;
        };
        let (bytes_part, desc_part) = after_colon.split_at(space);
        let Ok(bytes) = bytes_part.trim().parse::<usize>() else {
            continue;
        };
        nodes.push(MassifAllocationNode {
            bytes,
            function: desc_part.trim().to_string(),
            children: Vec::new(),
        });
    }
    nodes
}

/// Parse a Massif raw report. Errors when no `mem_heap_B=` field is found
/// anywhere -- the one field every real massif snapshot carries -- meaning
/// the file is not recognizable as massif output at all.
pub fn parse_massif_report(content: &str) -> Result<MassifReport> {
    if !content.contains("mem_heap_B=") {
        return Err(OptimError::InvalidConfig(
            "content does not look like a Massif report: no 'mem_heap_B=' field found".to_string(),
        ));
    }

    let mut timeline: Vec<(u64, usize)> = Vec::new();
    let mut pending_time: Option<u64> = None;
    let mut pending_heap: Option<usize> = None;
    let mut pending_extra: usize = 0;

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if let Some(value) = line.strip_prefix("time=") {
            if let (Some(t), Some(h)) = (pending_time.take(), pending_heap.take()) {
                timeline.push((t, h + pending_extra));
            }
            pending_extra = 0;
            pending_time = value.trim().parse().ok();
        } else if let Some(value) = line.strip_prefix("mem_heap_extra_B=") {
            pending_extra = value.trim().parse().unwrap_or(0);
        } else if let Some(value) = line.strip_prefix("mem_heap_B=") {
            pending_heap = value.trim().parse().ok();
        }
    }
    if let (Some(t), Some(h)) = (pending_time, pending_heap) {
        timeline.push((t, h + pending_extra));
    }

    let peak_memory_bytes = timeline.iter().map(|(_, mem)| *mem).max().unwrap_or(0);

    Ok(MassifReport {
        peak_memory_bytes,
        memory_timeline: timeline,
        allocation_tree: massif_tree_nodes(content),
    })
}

/// One leak-backtrace group from `heaptrack_print`'s summary, e.g. `"12
/// allocations with 4.00K (0.10%) total leaked in some_function:"` followed
/// by its indented backtrace frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaptrackLeakedAllocation {
    /// The group's combined leaked size, as reported by heaptrack (already
    /// an aggregate over every allocation sharing this backtrace -- this
    /// parser does not fan a group back out into per-allocation entries,
    /// since heaptrack's text summary does not expose per-allocation sizes).
    pub size_bytes: usize,
    pub call_stack: Vec<String>,
    /// heaptrack_print's text summary does not expose a per-allocation
    /// wall-clock timestamp. `0` means "not measured" (see module docs),
    /// never a fabricated value.
    pub allocation_time_ms: u64,
}

/// Summary statistics plus leak groups parsed from `heaptrack_print`'s text
/// output.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HeaptrackReport {
    /// From the `calls to allocation functions:` summary line.
    pub total_allocations: usize,
    /// From the `peak heap memory consumption:` summary line.
    pub peak_memory_bytes: usize,
    pub leaked_allocations: Vec<HeaptrackLeakedAllocation>,
}

fn parse_heaptrack_leak_header(line: &str, marker_idx: usize) -> Option<(usize, String)> {
    const MARKER: &str = " allocations with ";
    let after_marker = &line[marker_idx + MARKER.len()..];
    let size_token = after_marker.split_whitespace().next()?;
    let size_bytes = parse_byte_size(size_token)?;
    let function = after_marker
        .split("total leaked in")
        .nth(1)
        .map(|f| f.trim().trim_end_matches(':').trim().to_string())
        .filter(|f| !f.is_empty())
        .unwrap_or_else(|| "<unknown>".to_string());
    Some((size_bytes, function))
}

/// Parse `heaptrack_print`'s text summary (typically saved as e.g.
/// `heaptrack_analysis.txt`).
///
/// Errors when none of this format's recognizable markers are present --
/// `calls to allocation functions:`, `peak heap memory consumption:`, or an
/// ` allocations with ... total leaked in ` group header.
pub fn parse_heaptrack_report(content: &str) -> Result<HeaptrackReport> {
    let recognizable = content.contains("allocations with")
        || content.contains("calls to allocation functions")
        || content.contains("peak heap memory consumption");
    if !recognizable {
        return Err(OptimError::InvalidConfig(
            "content does not look like heaptrack_print output: none of the expected summary \
             markers ('calls to allocation functions:', 'peak heap memory consumption:', \
             '<N> allocations with <size> ... total leaked in ...:') were found"
                .to_string(),
        ));
    }

    let mut total_allocations = 0usize;
    let mut peak_memory_bytes = 0usize;
    let mut leaked = Vec::new();

    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();

        if let Some(rest) = line.strip_prefix("calls to allocation functions:") {
            if let Some(count_token) = rest.split_whitespace().next() {
                if let Ok(n) = count_token.replace(',', "").parse::<usize>() {
                    total_allocations = n;
                }
            }
            i += 1;
            continue;
        }
        if let Some(rest) = line.strip_prefix("peak heap memory consumption:") {
            if let Some(size_token) = rest.split_whitespace().next() {
                if let Some(bytes) = parse_byte_size(size_token) {
                    peak_memory_bytes = bytes;
                }
            }
            i += 1;
            continue;
        }
        if let Some(idx) = line.find(" allocations with ") {
            if let Some((size_bytes, function)) = parse_heaptrack_leak_header(line, idx) {
                let mut call_stack = Vec::new();
                let mut j = i + 1;
                while j < lines.len() {
                    let frame = lines[j].trim();
                    if frame.is_empty() || frame.contains(" allocations with ") {
                        break;
                    }
                    call_stack.push(frame.strip_prefix("at ").unwrap_or(frame).to_string());
                    j += 1;
                }
                leaked.push(HeaptrackLeakedAllocation {
                    size_bytes,
                    call_stack,
                    allocation_time_ms: 0,
                });
                let _ = function; // Reserved for a future per-site grouping key.
                i = j;
                continue;
            }
        }
        i += 1;
    }

    Ok(HeaptrackReport {
        total_allocations,
        peak_memory_bytes,
        leaked_allocations: leaked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_byte_size_units() {
        assert_eq!(parse_byte_size("512B"), Some(512));
        assert_eq!(parse_byte_size("512"), Some(512));
        assert_eq!(parse_byte_size("4.00K"), Some(4096));
        assert_eq!(parse_byte_size("48.00M"), Some(50331648));
        assert_eq!(parse_byte_size("1.00G"), Some(1073741824));
        assert_eq!(
            parse_byte_size("1.20MB"),
            Some((1.2_f64 * 1024.0 * 1024.0).round() as usize)
        );
        assert_eq!(parse_byte_size("not-a-size"), None);
        assert_eq!(parse_byte_size(""), None);
    }

    const VALGRIND_XML: &str = r#"<?xml version="1.0"?>
<valgrindoutput>
<protocolversion>4</protocolversion>
<error>
  <unique>0x1</unique>
  <kind>Leak_DefinitelyLost</kind>
  <xwhat>
    <text>512 bytes in 1 blocks are definitely lost</text>
    <leakedbytes>512</leakedbytes>
    <leakedblocks>1</leakedblocks>
  </xwhat>
  <stack>
    <frame><ip>0x1</ip><fn>malloc</fn></frame>
    <frame><ip>0x2</ip><fn>optimizer_alloc</fn></frame>
  </stack>
</error>
<error>
  <unique>0x2</unique>
  <kind>Leak_PossiblyLost</kind>
  <xwhat>
    <text>256 bytes in 2 blocks are possibly lost</text>
    <leakedbytes>256</leakedbytes>
    <leakedblocks>2</leakedblocks>
  </xwhat>
  <stack>
    <frame><ip>0x3</ip><fn>calloc</fn></frame>
  </stack>
</error>
</valgrindoutput>
"#;

    #[test]
    fn test_parse_valgrind_xml_extracts_real_leaks() {
        let report = parse_valgrind_xml(VALGRIND_XML).expect("valid valgrind xml parses");
        assert_eq!(report.total_leaks, 2);
        assert_eq!(report.leaked_bytes, 512 + 256);
        assert_eq!(report.leak_records[0].bytes_leaked, 512);
        assert_eq!(report.leak_records[0].blocks_leaked, 1);
        assert_eq!(report.leak_records[0].leak_kind, "definitely lost");
        assert_eq!(
            report.leak_records[0].call_stack,
            vec!["malloc".to_string(), "optimizer_alloc".to_string()]
        );
        assert_eq!(report.leak_records[1].leak_kind, "possibly lost");
        assert_eq!(report.leak_records[1].bytes_leaked, 256);
    }

    #[test]
    fn test_parse_valgrind_xml_clean_run_is_real_zero_not_error() {
        let clean = "<?xml version=\"1.0\"?>\n<valgrindoutput>\n<protocolversion>4</protocolversion>\n</valgrindoutput>\n";
        let report = parse_valgrind_xml(clean).expect("a valid, leak-free report still parses");
        assert_eq!(report.total_leaks, 0);
        assert_eq!(report.leaked_bytes, 0);
        assert!(report.leak_records.is_empty());
    }

    #[test]
    fn test_parse_valgrind_xml_rejects_unrecognizable_content() {
        // Regression (F54): must not silently return the old hardcoded
        // two-record fixture for a file that isn't Valgrind output at all.
        let result = parse_valgrind_xml("this is not xml, just some notes");
        assert!(result.is_err());
    }

    const MASSIF_REPORT: &str = "desc: --pages-as-heap=no --time-unit=B\n\
cmd: ./a.out\n\
time_unit: i\n\
#-----------\n\
snapshot=0\n\
#-----------\n\
time=0\n\
mem_heap_B=1048576\n\
mem_heap_extra_B=0\n\
mem_stacks_B=0\n\
heap_tree=empty\n\
#-----------\n\
snapshot=1\n\
#-----------\n\
time=2000000\n\
mem_heap_B=52428800\n\
mem_heap_extra_B=1024\n\
mem_stacks_B=0\n\
heap_tree=detailed\n\
n2: 52429824 (heap allocation functions) malloc/new/new[]\n\
 n1: 41943040 0x1000: adam_buffers (optimizer.c:10)\n\
 n1: 10485760 0x2000: gradient_buffers (optimizer.c:20)\n";

    #[test]
    fn test_parse_massif_report_extracts_real_timeline() {
        let report = parse_massif_report(MASSIF_REPORT).expect("valid massif report parses");
        assert_eq!(
            report.memory_timeline,
            vec![(0, 1048576), (2000000, 52428800 + 1024)]
        );
        assert_eq!(report.peak_memory_bytes, 52428800 + 1024);
        assert!(report
            .allocation_tree
            .iter()
            .any(|node| node.function.contains("adam_buffers") && node.bytes == 41943040));
    }

    #[test]
    fn test_parse_massif_report_rejects_unrecognizable_content() {
        assert!(parse_massif_report("just some unrelated text").is_err());
    }

    const HEAPTRACK_REPORT: &str = "total runtime: 12.34s.\n\
calls to allocation functions: 15420 (1234.5/s)\n\
peak heap memory consumption: 48.00M\n\
peak RSS (including heaptrack overhead): 52.00M\n\
\n\
12 allocations with 4.00K (0.10%) total leaked in temporary_buffer_alloc:\n\
    temporary_buffer_alloc\n\
    at optimizer.c:42\n\
    optimizer_step\n\
    at optimizer.c:100\n\
\n\
3 allocations with 512B (0.01%) total leaked in another_fn:\n\
    another_fn\n";

    #[test]
    fn test_parse_heaptrack_report_extracts_real_summary_and_leaks() {
        let report =
            parse_heaptrack_report(HEAPTRACK_REPORT).expect("valid heaptrack report parses");
        assert_eq!(report.total_allocations, 15420);
        assert_eq!(report.peak_memory_bytes, parse_byte_size("48.00M").unwrap());
        assert_eq!(report.leaked_allocations.len(), 2);
        assert_eq!(report.leaked_allocations[0].size_bytes, 4096);
        assert_eq!(
            report.leaked_allocations[0].call_stack,
            vec![
                "temporary_buffer_alloc".to_string(),
                "optimizer.c:42".to_string(),
                "optimizer_step".to_string(),
                "optimizer.c:100".to_string(),
            ]
        );
        assert_eq!(report.leaked_allocations[0].allocation_time_ms, 0);
        assert_eq!(report.leaked_allocations[1].size_bytes, 512);
    }

    #[test]
    fn test_parse_heaptrack_report_rejects_unrecognizable_content() {
        assert!(parse_heaptrack_report("nothing heaptrack-shaped here").is_err());
    }
}
