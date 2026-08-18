//! Memory usage reporter binary.
//!
//! Emits a report of the current process' real memory usage, measured via
//! [`optirs_bench::system_sampler::SystemSampler`] (backed by `sysinfo`).
//! It reports resident set size (RSS) and virtual memory -- the figures the
//! operating system actually accounts for -- rather than a fabricated
//! heap/stack split, which is not portably observable.

use std::env;
use std::fs::File;
use std::io::{self, Write};

use optirs_bench::system_sampler::SystemSampler;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <output_file>", args[0]);
        std::process::exit(1);
    }

    let output_file = &args[1];
    if let Err(e) = generate_memory_report(output_file) {
        eprintln!("Failed to generate memory report: {e}");
        std::process::exit(1);
    }

    println!("Memory report generated: {output_file}");
    Ok(())
}

fn generate_memory_report(output_file: &str) -> io::Result<()> {
    let sampler =
        SystemSampler::new().map_err(|e| io::Error::other(format!("sampler init failed: {e}")))?;
    let process = sampler
        .sample_process()
        .map_err(|e| io::Error::other(format!("process sample failed: {e}")))?;
    let system = sampler.sample_system();

    let mut file = File::create(output_file)?;

    writeln!(file, "Memory Usage Report")?;
    writeln!(file, "==================")?;
    writeln!(file)?;
    writeln!(file, "Process Memory Usage:")?;
    writeln!(
        file,
        "- Resident (RSS): {} bytes ({:.2} MiB)",
        process.rss_bytes,
        process.rss_bytes as f64 / (1024.0 * 1024.0)
    )?;
    writeln!(
        file,
        "- Virtual:        {} bytes ({:.2} MiB)",
        process.virtual_bytes,
        process.virtual_bytes as f64 / (1024.0 * 1024.0)
    )?;
    writeln!(file)?;
    writeln!(file, "System Memory:")?;
    writeln!(
        file,
        "- Total:     {} bytes ({:.2} MiB)",
        system.total_memory_bytes,
        system.total_memory_bytes as f64 / (1024.0 * 1024.0)
    )?;
    writeln!(
        file,
        "- Used:      {} bytes ({:.2} MiB)",
        system.used_memory_bytes,
        system.used_memory_bytes as f64 / (1024.0 * 1024.0)
    )?;
    writeln!(
        file,
        "- Available: {} bytes ({:.2} MiB)",
        system.available_memory_bytes,
        system.available_memory_bytes as f64 / (1024.0 * 1024.0)
    )?;

    Ok(())
}
