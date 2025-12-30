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

    // Check for common security patterns
    if file_name.ends_with(".rs") {
        if let Ok(content) = fs::read_to_string(path) {
            // Check for unsafe blocks
            if content.contains("unsafe") {
                issues.push(format!("Unsafe code found in: {}", path.display()));
            }

            // Check for unwrap() calls
            if content.contains(".unwrap()") {
                issues.push(format!(
                    "Potential panic with unwrap() in: {}",
                    path.display()
                ));
            }

            // Check for debug prints
            if content.contains("println!") && content.contains("password") {
                issues.push(format!("Potential password logging in: {}", path.display()));
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
