//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::regression_tester::distributions;
use std::f64::consts::PI;

use super::types::{
    AdvancedMemoryPattern, AdvancedPatternConfig, AdvancedPatternType, FFTProcessor,
    HypothesisTestEngine, StatisticalProperties, TrendInfo, WaveletProcessor,
};
use super::types_7::{
    AdvancedPatternDetector, FrequencyCharacteristics, PatternClassifier, PatternEvolution,
};

/// Current Unix timestamp in seconds, or `0` if the system clock is set before
/// the Unix epoch. Non-panicking replacement for `.expect(...)` on
/// `SystemTime::duration_since`.
pub(super) fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// Average second derivative ("slope of slopes") of a series, computed as the
/// OLS slope of its first differences.
///
/// Returns `0.0` for series too short to define curvature. A straight line has
/// constant first differences and therefore zero acceleration, as expected.
pub(super) fn calculate_acceleration(data: &[f64]) -> f64 {
    if data.len() < 3 {
        return 0.0;
    }
    let diffs: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
    if let Some(trend) = distributions::linear_regression(&diffs) {
        trend.slope
    } else {
        // Fewer than three differences: fall back to the mean second difference.
        let second: Vec<f64> = diffs.windows(2).map(|w| w[1] - w[0]).collect();
        if second.is_empty() {
            0.0
        } else {
            second.iter().sum::<f64>() / second.len() as f64
        }
    }
}

/// Dominant spectral peak of a real signal.
///
/// Returns `(normalized_frequency_in_cycles_per_sample, prominence)` where the
/// prominence is the peak bin's share of the total non-DC magnitude. `None` when
/// the FFT is undefined (empty / non-finite input) or the signal carries no
/// alternating-current energy (e.g. a constant series).
pub(super) fn dominant_spectral_peak(data: &[f64]) -> Option<(f64, f64)> {
    let spectrum = distributions::magnitude_spectrum(data)?;
    if spectrum.len() < 2 {
        return None;
    }
    // Exclude the DC bin (index 0): it only reflects the mean level.
    let alternating = &spectrum[1..];
    let total: f64 = alternating.iter().map(|value| value.abs()).sum();
    if !(total > 0.0) {
        return None;
    }
    let (peak_offset, peak_magnitude) = alternating
        .iter()
        .enumerate()
        .max_by(|a, b| distributions::compare_finite(a.1, b.1))?;
    let peak_bin = peak_offset + 1;
    // padded_length == 2 * spectrum.len(); frequency = bin / padded_length.
    let normalized_frequency = peak_bin as f64 / (2.0 * spectrum.len() as f64);
    Some((normalized_frequency, *peak_magnitude / total))
}

/// Build a memory pattern from a statistically significant Mann-Kendall trend.
///
/// The direction is taken from the OLS slope: an increasing series is reported
/// as a `LeakSignature` (the memory-relevant case) and a decreasing series as
/// `LinearGrowth` with a negative slope. Returns `None` when the regression is
/// undefined (fewer than three finite points).
pub(super) fn build_mann_kendall_pattern(
    data: &[f64],
    p_value: f64,
) -> Option<AdvancedMemoryPattern> {
    let trend = distributions::linear_regression(data)?;
    let confidence = (1.0 - p_value).clamp(0.0, 1.0);
    let acceleration = calculate_acceleration(data);
    let pattern_type = if trend.slope >= 0.0 {
        AdvancedPatternType::LeakSignature {
            leak_rate: trend.slope,
            leak_acceleration: acceleration,
            leak_confidence: confidence,
        }
    } else {
        AdvancedPatternType::LinearGrowth {
            slope: trend.slope,
            intercept: trend.intercept,
            r_squared: trend.r_squared,
        }
    };

    Some(AdvancedMemoryPattern {
        id: "mann_kendall_trend".to_string(),
        pattern_type,
        confidence,
        signature: vec![trend.slope, trend.intercept, trend.r_squared, p_value],
        description: format!(
            "Mann-Kendall significant trend (p = {:.4}, slope = {:.4})",
            p_value, trend.slope
        ),
        frequency_characteristics: FrequencyCharacteristics::default(),
        statistical_properties: StatisticalProperties::default(),
        anomaly_score: 0.0,
        strength: confidence,
        periodicity: None,
        trend: TrendInfo {
            direction: trend.slope.signum(),
            strength: confidence,
            acceleration,
            stability: 0.0,
            change_points: Vec::new(),
        },
        leak_indicators: Vec::new(),
        evolution: PatternEvolution::default(),
    })
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    fn sinusoid(length: usize, cycles: f64) -> Vec<f64> {
        (0..length)
            .map(|i| (2.0 * PI * cycles * i as f64 / length as f64).sin())
            .collect()
    }

    // ---- F70: real FFT magnitude spectrum -------------------------------

    #[test]
    fn fft_spectrum_peaks_at_sinusoid_frequency() {
        // A pure sinusoid at an integer bin should peak at exactly that bin.
        let bin = 8usize;
        let signal = sinusoid(64, bin as f64);
        let processor = FFTProcessor::new();
        let spectrum = processor
            .compute_fft(&signal)
            .expect("compute_fft returns a spectrum");

        assert_eq!(spectrum.len(), 32, "64-sample signal -> 32 magnitude bins");
        let (peak_bin, _) = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| distributions::compare_finite(a.1, b.1))
            .expect("non-empty spectrum");
        assert_eq!(peak_bin, bin, "FFT peak must land on the sinusoid's bin");
    }

    #[test]
    fn wavelet_analysis_returns_real_coefficients() {
        let signal = sinusoid(64, 4.0);
        let processor = WaveletProcessor::new();
        let patterns = processor
            .analyze_wavelets(&signal)
            .expect("analyze_wavelets succeeds");

        assert!(
            !patterns.is_empty(),
            "the wavelet transform should find structure in a sinusoid"
        );
        let signature = &patterns[0].signature;
        assert!(!signature.is_empty(), "per-level energies must be reported");
        assert!(
            signature.iter().all(|value| value.is_finite()),
            "all wavelet-derived energies must be finite"
        );
        assert!(
            signature.iter().any(|&value| value > 0.0),
            "detail energy must be non-zero for a real oscillation"
        );
    }

    // ---- F71: real Mann-Kendall p-value ---------------------------------

    #[test]
    fn mann_kendall_p_value_separates_trend_from_noise() {
        let engine = HypothesisTestEngine::new();

        let increasing: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let p_increasing = engine
            .mann_kendall_test(&increasing)
            .expect("defined for a monotone series");
        assert!(
            p_increasing < 0.01,
            "monotonic increase -> tiny p-value, got {p_increasing}"
        );

        // A constant series has zero variance: the test is honestly undefined.
        let flat = vec![5.0; 40];
        assert!(
            engine.mann_kendall_test(&flat).is_none(),
            "a constant series yields no defined trend p-value"
        );

        // A stationary repeating cycle has no monotone trend -> large p-value.
        let stationary: Vec<f64> = (0..40).map(|i| ((i * 2) % 5) as f64).collect();
        if let Some(p_stationary) = engine.mann_kendall_test(&stationary) {
            assert!(
                p_stationary > 0.05,
                "stationary series -> large p-value, got {p_stationary}"
            );
        }
    }

    // ---- F72: deterministic classifier ----------------------------------

    #[test]
    fn classifier_detects_clear_linear_trend() {
        let classifier = PatternClassifier::new();
        let data: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let features = vec![0.0; 10];
        let patterns = classifier
            .classify_patterns(&features, &data)
            .expect("classification succeeds");

        assert!(
            !patterns.is_empty(),
            "a clear linear trend must be classified"
        );
        assert!(
            matches!(
                patterns[0].pattern_type,
                AdvancedPatternType::LinearGrowth { .. }
            ),
            "a ramp must be classified as linear growth"
        );
        assert!(patterns[0].confidence > 0.5);
    }

    #[test]
    fn classifier_detects_clear_periodic_signal() {
        let classifier = PatternClassifier::new();
        let data = sinusoid(64, 4.0);
        let features = vec![0.0; 10];
        let patterns = classifier
            .classify_patterns(&features, &data)
            .expect("classification succeeds");

        assert!(
            !patterns.is_empty(),
            "a clear periodic signal must be classified"
        );
        assert!(
            matches!(
                patterns[0].pattern_type,
                AdvancedPatternType::Periodic { .. }
            ),
            "a sinusoid must be classified as periodic"
        );
    }

    #[test]
    fn classifier_stays_silent_on_a_featureless_signal() {
        let classifier = PatternClassifier::new();
        // Constant series: no trend, no spectral peak -> no confident class.
        let data = vec![3.0; 32];
        let features = vec![0.0; 10];
        let patterns = classifier
            .classify_patterns(&features, &data)
            .expect("classification succeeds");
        assert!(
            patterns.is_empty(),
            "a constant signal is not a confident pattern"
        );
    }

    // ---- F74: no panic on empty / degenerate input ----------------------

    #[test]
    fn detect_patterns_handles_empty_and_degenerate_input() {
        let mut detector = AdvancedPatternDetector::new(AdvancedPatternConfig::default())
            .expect("detector constructs");

        // Empty and too-short inputs return no patterns without panicking.
        assert!(detector
            .detect_patterns(&[])
            .expect("no panic on empty input")
            .is_empty());
        assert!(detector
            .detect_patterns(&[1.0, 2.0])
            .expect("no panic on short input")
            .is_empty());

        // Long degenerate / non-finite inputs must not panic either.
        let constant = vec![3.0; 64];
        let _ = detector
            .detect_patterns(&constant)
            .expect("no panic on constant input");
        let with_nan = vec![f64::NAN; 64];
        let _ = detector
            .detect_patterns(&with_nan)
            .expect("no panic on NaN input");
    }

    #[test]
    fn acceleration_is_zero_for_a_line_and_positive_for_curvature() {
        let line: Vec<f64> = (0..10).map(|i| 2.0 * i as f64).collect();
        assert!(
            calculate_acceleration(&line).abs() < 1e-9,
            "a straight line has zero acceleration"
        );

        let accelerating: Vec<f64> = (0..10).map(|i| (i * i) as f64).collect();
        assert!(
            calculate_acceleration(&accelerating) > 0.0,
            "a convex series has positive acceleration"
        );
    }
}
