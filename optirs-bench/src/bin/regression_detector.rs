use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <baseline_file> <current_file>", args[0]);
        std::process::exit(1);
    }

    let baseline_file = &args[1];
    let current_file = &args[2];

    let regressions = detect_regressions(baseline_file, current_file)?;

    if regressions.is_empty() {
        println!("No performance regressions detected.");
    } else {
        println!("Performance regressions detected:");
        for regression in regressions {
            println!("- {}", regression);
        }
    }

    Ok(())
}

fn detect_regressions(baseline_file: &str, current_file: &str) -> io::Result<Vec<String>> {
    let mut regressions = Vec::new();

    let baseline_metrics = load_metrics(baseline_file)?;
    let current_metrics = load_metrics(current_file)?;

    for (test_name, baseline_time) in baseline_metrics {
        if let Some(current_time) = current_metrics.get(&test_name) {
            let threshold = 0.05; // 5% threshold
            let regression_ratio = (*current_time - baseline_time) / baseline_time;

            if regression_ratio > threshold {
                regressions.push(format!(
                    "{}: {:.2}% slower ({:.2}ms -> {:.2}ms)",
                    test_name,
                    regression_ratio * 100.0,
                    baseline_time,
                    current_time
                ));
            }
        }
    }

    Ok(regressions)
}

fn load_metrics(file_path: &str) -> io::Result<std::collections::HashMap<String, f64>> {
    let mut metrics = std::collections::HashMap::new();
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if let Some((test_name, time_str)) = line.split_once(':') {
            if let Ok(time) = time_str.trim().parse::<f64>() {
                metrics.insert(test_name.trim().to_string(), time);
            }
        }
    }

    Ok(metrics)
}
