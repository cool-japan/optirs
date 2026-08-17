use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <directory>", args[0]);
        std::process::exit(1);
    }

    let directory = &args[1];

    if !Path::new(directory).exists() {
        eprintln!("Error: Directory '{}' does not exist", directory);
        std::process::exit(1);
    }

    println!("Scanning directory: {}", directory);

    let vulnerabilities = scan_directory(directory);

    if vulnerabilities.is_empty() {
        println!("No security vulnerabilities detected.");
    } else {
        println!("Security vulnerabilities detected:");
        for vulnerability in vulnerabilities {
            println!("- {}", vulnerability);
        }
    }
}

fn scan_directory(directory: &str) -> Vec<String> {
    let mut vulnerabilities = Vec::new();

    if let Ok(entries) = fs::read_dir(directory) {
        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_file() {
                if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    vulnerabilities.extend(scan_file(&path, file_name));
                }
            } else if path.is_dir() {
                if let Some(dir_name) = path.to_str() {
                    vulnerabilities.extend(scan_directory(dir_name));
                }
            }
        }
    }

    vulnerabilities
}

fn scan_file(path: &Path, file_name: &str) -> Vec<String> {
    let mut issues = Vec::new();

    // Check for common security patterns, line by line, so every finding
    // carries a real location and comments do not trigger false positives.
    if file_name.ends_with(".rs") {
        if let Ok(content) = fs::read_to_string(path) {
            for (index, line) in content.lines().enumerate() {
                let line_number = index + 1;
                // Ignore anything after a `//` line comment: findings that live
                // only in comments are not real code issues.
                let code = line.split("//").next().unwrap_or(line);

                if contains_keyword(code, "unsafe") {
                    issues.push(format!("Unsafe code at {}:{}", path.display(), line_number));
                }

                if code.contains(".unwrap()") {
                    issues.push(format!(
                        "Potential panic with unwrap() at {}:{}",
                        path.display(),
                        line_number
                    ));
                }

                // Password logging: the print/log call and the word "password"
                // must appear on the same line to be reported.
                let lowered = code.to_ascii_lowercase();
                let logs = [
                    "println!",
                    "print!",
                    "eprintln!",
                    "eprint!",
                    "log::",
                    "debug!",
                    "info!",
                ]
                .iter()
                .any(|m| code.contains(m));
                if logs && lowered.contains("password") {
                    issues.push(format!(
                        "Potential password logging at {}:{}",
                        path.display(),
                        line_number
                    ));
                }
            }
        }
    }

    // Check for sensitive file patterns
    if file_name.contains("secret") || file_name.contains("password") || file_name.contains(".key")
    {
        issues.push(format!("Potentially sensitive file: {}", path.display()));
    }

    issues
}

/// Return true when `keyword` appears in `text` as a standalone identifier
/// (bounded by non-identifier characters), avoiding matches inside longer
/// words such as `unsafely` or `get_unsafe_ptr`.
fn contains_keyword(text: &str, keyword: &str) -> bool {
    let bytes = text.as_bytes();
    let mut start = 0;
    while let Some(pos) = text[start..].find(keyword) {
        let abs = start + pos;
        let before_ok = abs == 0 || !is_ident_byte(bytes[abs - 1]);
        let after_index = abs + keyword.len();
        let after_ok = after_index >= bytes.len() || !is_ident_byte(bytes[after_index]);
        if before_ok && after_ok {
            return true;
        }
        start = abs + keyword.len();
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
