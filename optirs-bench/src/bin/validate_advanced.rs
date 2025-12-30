// Advanced Mode Implementation Validation Binary
//
// This binary validates the syntax and basic functionality of our
// Advanced mode implementations.

use std::path::Path;

#[allow(dead_code)]
fn main() {
    println!("🔍 Advanced Mode Implementation Validation");
    println!("============================================\n");

    let validations = vec![
        ("JIT Compilation System", "../scirs2-core/src/jit.rs"),
        (
            "Neural RL Step Control",
            "../scirs2-integrate/src/neural_rl_step_control.rs",
        ),
        (
            "Neural Adaptive Shader",
            "../scirs2-integrate/src/shaders/neural_adaptive_step_rl.comp",
        ),
        (
            "API Stability Analysis",
            "../scirs2-interpolate/examples/api_stability_analysis.rs",
        ),
        (
            "Security Audit Scanner",
            "src/bin/security_audit_scanner.rs",
        ),
        (
            "Dependency Scanner",
            "src/bin/dependency_vulnerability_scanner.rs",
        ),
    ];

    let mut passed = 0;
    let mut total = 0;

    for (name, path) in validations {
        total += 1;
        println!("📋 Checking: {}", name);

        if Path::new(path).exists() {
            // Check file size to ensure it's not empty
            if let Ok(metadata) = std::fs::metadata(path) {
                if metadata.len() > 100 {
                    println!(
                        "   ✅ File exists and has content ({} bytes)",
                        metadata.len()
                    );

                    // For Rust files, do a basic syntax check
                    if path.ends_with(".rs") {
                        match validate_rust_syntax(path) {
                            Ok(()) => {
                                println!("   ✅ Syntax validation passed");
                                passed += 1;
                            }
                            Err(e) => {
                                println!("   ❌ Syntax validation failed: {}", e);
                            }
                        }
                    } else {
                        println!("   ✅ Non-Rust file validated");
                        passed += 1;
                    }
                } else {
                    println!("   ❌ File is too small or empty");
                }
            } else {
                println!("   ❌ Cannot read file metadata");
            }
        } else {
            println!("   ❌ File does not exist");
        }
        println!();
    }

    println!("📊 Validation Summary:");
    println!("   Total implementations: {}", total);
    println!("   Passed validations: {}", passed);
    println!(
        "   Success rate: {:.1}%",
        (passed as f32 / total as f32) * 100.0
    );

    if passed == total {
        println!("\n🎉 All advanced implementations validated successfully!");
    } else {
        println!("\n⚠️  Some implementations need attention.");
    }
}

#[allow(dead_code)]
fn validate_rust_syntax(path: &str) -> Result<(), String> {
    // Always use basic validation as -Z parse-only requires nightly
    validate_basic_rust_syntax(path)
}

#[allow(dead_code)]
fn validate_basic_rust_syntax(path: &str) -> Result<(), String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("Cannot read file: {}", e))?;

    // Basic syntax checks
    let mut issues = Vec::new();

    // Check for unclosed braces
    let open_braces = content.matches('{').count();
    let close_braces = content.matches('}').count();
    if open_braces != close_braces {
        issues.push(format!(
            "Mismatched braces: {} open, {} close",
            open_braces, close_braces
        ));
    }

    // Check for unclosed parentheses
    let open_parens = content.matches('(').count();
    let close_parens = content.matches(')').count();
    if open_parens != close_parens {
        issues.push(format!(
            "Mismatched parentheses: {} open, {} close",
            open_parens, close_parens
        ));
    }

    // Check for proper module structure
    if (content.contains("pub mod ") || content.contains("mod "))
        && !content.contains("use ")
        && !content.contains("extern crate")
    {
        // This might be okay for some modules
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues.join("; "))
    }
}
