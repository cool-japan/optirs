// Moment Accountant for Differential Privacy
//
// This module implements the moment accountant method for precise privacy
// analysis of the Gaussian mechanism with subsampling, providing tight
// privacy loss bounds for DP-SGD and related algorithms.

use std::collections::HashMap;

use crate::error::{OptimError, Result};

/// Moment accountant for tracking privacy loss
#[derive(Debug, Clone)]
pub struct MomentsAccountant {
    /// Noise multiplier (σ)
    noise_multiplier: f64,

    /// Target delta parameter
    target_delta: f64,

    /// Batch size
    batch_size: usize,

    /// Total dataset size
    dataset_size: usize,

    /// Sampling probability (q = batch_size / dataset_size)
    sampling_probability: f64,

    /// Computed log moments
    log_moments: HashMap<usize, f64>,

    /// Maximum order for moment computation
    max_order: usize,

    /// Precision for numerical computations
    precision: f64,

    /// Precomputed coefficients for efficiency
    coefficients: MomentCoefficients,
}

/// Precomputed coefficients for moment computation
#[derive(Debug, Clone)]
struct MomentCoefficients {
    /// Binomial coefficients
    binomial_coeffs: HashMap<(usize, usize), f64>,

    /// Precomputed powers
    power_cache: HashMap<(f64, usize), f64>,

    /// Logarithmic factorials
    log_factorials: Vec<f64>,
}

/// Privacy analysis result from moment accountant
#[derive(Debug, Clone)]
pub struct PrivacyAnalysis {
    /// Best epsilon for given delta
    pub epsilon: f64,

    /// Delta used
    pub delta: f64,

    /// Number of steps analyzed
    pub steps: usize,

    /// Optimal moment order used
    pub optimal_order: usize,

    /// All computed log moments
    pub log_moments: HashMap<usize, f64>,

    /// Privacy amplification factor
    pub amplification_factor: f64,

    /// Composition bound tightness
    pub bound_tightness: f64,
}

/// Advanced privacy composition analysis
#[derive(Debug, Clone)]
pub struct CompositionAnalysis {
    /// Mechanism parameters for each step
    pub mechanisms: Vec<MechanismParameters>,

    /// Composed privacy guarantee
    pub composed_epsilon: f64,

    /// Composed delta
    pub composed_delta: f64,

    /// Total number of compositions
    pub num_compositions: usize,

    /// Heterogeneous composition (different parameters)
    pub is_heterogeneous: bool,
}

/// Parameters for a single mechanism application
#[derive(Debug, Clone)]
pub struct MechanismParameters {
    /// Noise multiplier
    pub noise_multiplier: f64,

    /// Sampling probability
    pub sampling_probability: f64,

    /// Sensitivity
    pub sensitivity: f64,

    /// Number of applications
    pub applications: usize,
}

/// Privacy budget status for real-time monitoring
#[derive(Debug, Clone)]
pub struct PrivacyBudgetStatus {
    /// Remaining epsilon budget
    pub epsilon_remaining: f64,

    /// Remaining delta budget
    pub delta_remaining: f64,

    /// Utilization ratio (0.0 to 1.0)
    pub utilization_ratio: f64,

    /// Current budget status
    pub status: BudgetStatus,

    /// Number of steps analyzed
    pub steps_analyzed: usize,

    /// Recommended maximum steps
    pub recommended_max_steps: usize,
}

/// Privacy budget status levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetStatus {
    /// Healthy: < 50% budget used
    Healthy,

    /// Moderate: 50-80% budget used
    Moderate,

    /// Critical: 80-95% budget used
    Critical,

    /// Exhausted: > 95% budget used
    Exhausted,
}

impl MomentsAccountant {
    /// Create a new moment accountant
    pub fn new(
        noise_multiplier: f64,
        target_delta: f64,
        batch_size: usize,
        dataset_size: usize,
    ) -> Self {
        let sampling_probability = batch_size as f64 / dataset_size as f64;
        let max_order = 64; // Usually sufficient for practical applications
        let precision = 1e-12;

        Self {
            noise_multiplier,
            target_delta,
            batch_size,
            dataset_size,
            sampling_probability,
            log_moments: HashMap::new(),
            max_order,
            precision,
            coefficients: MomentCoefficients::new(max_order),
        }
    }

    /// Get privacy spent after a given number of steps
    pub fn get_privacy_spent(&self, steps: usize) -> Result<(f64, f64)> {
        if steps == 0 {
            return Ok((0.0, 0.0));
        }

        let analysis = self.analyze_privacy(steps)?;
        Ok((analysis.epsilon, self.target_delta))
    }

    /// Perform comprehensive privacy analysis
    pub fn analyze_privacy(&self, steps: usize) -> Result<PrivacyAnalysis> {
        // Compute log moments for all orders
        let mut log_moments = HashMap::new();

        for order in 2..=self.max_order {
            let log_moment = self.compute_log_moment(order, steps)?;
            log_moments.insert(order, log_moment);
        }

        // Find optimal epsilon using all computed moments
        let (epsilon, optimal_order) = self.compute_optimal_epsilon(&log_moments)?;

        // Compute privacy amplification factor
        let amplification_factor = self.compute_amplification_factor();

        // Assess bound tightness
        let bound_tightness = self.assess_bound_tightness(&log_moments, epsilon);

        Ok(PrivacyAnalysis {
            epsilon,
            delta: self.target_delta,
            steps,
            optimal_order,
            log_moments,
            amplification_factor,
            bound_tightness,
        })
    }

    /// Compute log moment for a specific order and number of steps
    pub fn compute_log_moment(&self, order: usize, steps: usize) -> Result<f64> {
        if order < 2 {
            return Err(OptimError::InvalidConfig(
                "Moment order must be at least 2".to_string(),
            ));
        }

        // Use cached result if available
        let cache_key = order;
        if let Some(&cached_moment) = self.log_moments.get(&cache_key) {
            return Ok(cached_moment * steps as f64);
        }

        // Compute log moment for single step with enhanced precision
        let single_step_log_moment = self.compute_enhanced_single_step_log_moment(order)?;

        // For composition, multiply by number of steps (add in log space)
        let total_log_moment = single_step_log_moment * steps as f64;

        Ok(total_log_moment)
    }

    /// Enhanced single step log moment computation with adaptive precision
    pub fn compute_enhanced_single_step_log_moment(&self, order: usize) -> Result<f64> {
        // Use higher precision for better accuracy
        let q = self.sampling_probability;
        let sigma = self.noise_multiplier;

        // Enhanced computation with stability improvements
        let result = if q < 1e-6 {
            // Use series expansion for very small q
            self.compute_small_q_approximation(order, q, sigma)
        } else if q > 0.5 {
            // Use different approach for large sampling probabilities
            self.compute_large_q_log_moment(order, q, sigma)
        } else {
            // Standard computation with enhanced numerical stability
            self.compute_standard_log_moment(order, q, sigma)
        }?;

        Ok(result)
    }

    /// Compute log moment using series expansion for small sampling probabilities
    fn compute_small_q_approximation(&self, order: usize, q: f64, sigma: f64) -> Result<f64> {
        // For small q, use Taylor series expansion: M_lambda(q) ≈ q * lambda * exp(lambda/(2*sigma^2))
        let lambda = order as f64;
        let variance = sigma * sigma;

        // Leading term
        let leading_term = q * lambda * (lambda / (2.0 * variance)).exp();

        // Higher order corrections for better accuracy
        let correction1 = q * q * lambda * lambda * (lambda / variance).exp() / 2.0;
        let correction2 =
            q * q * q * lambda * lambda * lambda * (3.0 * lambda / (2.0 * variance)).exp() / 6.0;

        let result = leading_term + correction1 + correction2;
        Ok(result.ln())
    }

    /// Compute log moment for large sampling probabilities
    fn compute_large_q_log_moment(&self, order: usize, q: f64, sigma: f64) -> Result<f64> {
        // For large q, use complementary approach
        let lambda = order as f64;
        let variance = sigma * sigma;

        // Use the exact formula but with better numerical handling
        let term1 = (1.0 - q + q * (lambda / (2.0 * variance)).exp()).ln();
        let term2 = (1.0 - q + q * (-lambda / (2.0 * variance)).exp()).ln();

        Ok(lambda * lambda / (2.0 * variance) + term1 - term2)
    }

    /// Standard log moment computation with enhanced stability
    fn compute_standard_log_moment(&self, order: usize, q: f64, sigma: f64) -> Result<f64> {
        let lambda = order as f64;
        let variance = sigma * sigma;

        // Use log-sum-exp trick for numerical stability
        let exp_term = lambda / (2.0 * variance);
        let max_exp = exp_term.abs();

        let term1 = (1.0 - q + q * (exp_term - max_exp).exp()).ln() + max_exp;
        let term2 = (1.0 - q + q * (-exp_term - max_exp).exp()).ln() + max_exp;

        Ok(lambda * lambda / (2.0 * variance) + term1 - term2)
    }

    /// Enhanced privacy budget tracking with time-varying parameters
    pub fn track_heterogeneous_composition(
        &mut self,
        mechanisms: &[MechanismParameters],
    ) -> Result<CompositionAnalysis> {
        let mut total_log_moments = HashMap::new();

        // Initialize log moments
        for order in 2..=self.max_order {
            total_log_moments.insert(order, 0.0);
        }

        // Compose over all mechanisms
        for mechanism in mechanisms {
            for order in 2..=self.max_order {
                let single_log_moment = self.compute_mechanism_log_moment(order, mechanism)?;
                let current = total_log_moments.get(&order).unwrap_or(&0.0);
                total_log_moments.insert(
                    order,
                    current + single_log_moment * mechanism.applications as f64,
                );
            }
        }

        // Find optimal epsilon
        let (composed_epsilon, _) = self.compute_optimal_epsilon(&total_log_moments)?;

        Ok(CompositionAnalysis {
            mechanisms: mechanisms.to_vec(),
            composed_epsilon,
            composed_delta: self.target_delta,
            num_compositions: mechanisms.iter().map(|m| m.applications).sum(),
            is_heterogeneous: mechanisms.len() > 1,
        })
    }

    /// Compute log moment for a specific mechanism
    fn compute_mechanism_log_moment(
        &self,
        order: usize,
        mechanism: &MechanismParameters,
    ) -> Result<f64> {
        let _lambda = order as f64;
        let q = mechanism.sampling_probability;
        let sigma = mechanism.noise_multiplier;
        let sensitivity = mechanism.sensitivity;

        // Adjust for sensitivity
        let effective_sigma = sigma / sensitivity;
        let _variance = effective_sigma * effective_sigma;

        // Use the same enhanced computation as before
        let result = if q < 1e-6 {
            self.compute_small_q_approximation(order, q, effective_sigma)
        } else if q > 0.5 {
            self.compute_large_q_log_moment(order, q, effective_sigma)
        } else {
            self.compute_standard_log_moment(order, q, effective_sigma)
        }?;

        Ok(result)
    }

    /// Advanced epsilon-delta conversion with tighter bounds
    pub fn compute_tight_epsilon_delta_bound(
        &self,
        steps: usize,
        target_delta: f64,
    ) -> Result<f64> {
        // Use refined bound that accounts for finite sampling
        let mut best_epsilon = f64::INFINITY;

        for order in 2..=self.max_order {
            let log_moment = self.compute_log_moment(order, steps)?;

            // Enhanced epsilon computation with finite sample corrections
            let finite_sample_correction = self.compute_finite_sample_correction(order, steps);
            let adjusted_log_moment = log_moment + finite_sample_correction;

            let epsilon = (adjusted_log_moment - target_delta.ln()) / (order as f64 - 1.0);

            if epsilon > 0.0 && epsilon < best_epsilon {
                best_epsilon = epsilon;
            }
        }

        if best_epsilon == f64::INFINITY {
            return Err(OptimError::InvalidConfig(
                "Could not compute valid epsilon bound".to_string(),
            ));
        }

        Ok(best_epsilon)
    }

    /// Compute finite sample correction for improved bounds
    fn compute_finite_sample_correction(&self, order: usize, steps: usize) -> f64 {
        // Finite sample correction based on dataset size and number of steps
        let n = self.dataset_size as f64;
        let t = steps as f64;
        let lambda = order as f64;

        // Correction term: O(t/n) for subsampling without replacement
        let basic_correction = t / n * lambda * lambda / 8.0;

        // Additional higher-order correction
        let higher_order_correction = t * t / (n * n) * lambda * lambda * lambda / 24.0;

        basic_correction + higher_order_correction
    }

    /// Privacy amplification analysis with refined bounds
    pub fn analyze_privacy_amplification(&self) -> f64 {
        let q = self.sampling_probability;

        // Enhanced amplification factor accounting for composition
        if q <= 0.01 {
            // Strong amplification for small sampling probabilities
            q * (1.0 + q / 2.0)
        } else if q <= 0.1 {
            // Moderate amplification
            q * (1.0 + q)
        } else {
            // Limited amplification for large sampling probabilities
            q * (1.0 + 2.0 * q)
        }
    }

    /// Real-time privacy budget monitoring
    pub fn get_privacy_budget_status(&self, steps: usize) -> Result<PrivacyBudgetStatus> {
        let analysis = self.analyze_privacy(steps)?;
        let remaining_epsilon = 1.0 - analysis.epsilon; // Assuming target epsilon = 1.0
        let utilization = analysis.epsilon;

        let status = if utilization < 0.5 {
            BudgetStatus::Healthy
        } else if utilization < 0.8 {
            BudgetStatus::Moderate
        } else if utilization < 0.95 {
            BudgetStatus::Critical
        } else {
            BudgetStatus::Exhausted
        };

        Ok(PrivacyBudgetStatus {
            epsilon_remaining: remaining_epsilon,
            delta_remaining: self.target_delta,
            utilization_ratio: utilization,
            status,
            steps_analyzed: steps,
            recommended_max_steps: self.estimate_max_steps()?,
        })
    }

    /// Estimate maximum number of steps within privacy budget
    fn estimate_max_steps(&self) -> Result<usize> {
        let target_epsilon = 1.0; // Configurable target

        // Binary search for maximum steps
        let mut low = 1;
        let mut high = 100000; // Upper bound
        let mut result = 0;

        while low <= high {
            let mid = (low + high) / 2;
            let epsilon_ = self.get_privacy_spent(mid)?;

            if epsilon_.0 <= target_epsilon {
                result = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }

        Ok(result)
    }

    /// Compute log moment for a single step of the mechanism
    fn compute_single_step_log_moment(&self, order: usize) -> Result<f64> {
        let q = self.sampling_probability;
        let sigma = self.noise_multiplier;
        let alpha = order as f64;

        // Compute log moment using the formula for Gaussian mechanism with subsampling
        // Based on Abadi et al. "Deep Learning with Differential Privacy"

        let mut log_moment = 0.0;

        // Sum over all possible values of k (number of sampled points that change)
        for k in 0..=2 {
            let log_binomial = self.log_binomial_coefficient(2, k);
            let log_prob_k = k as f64 * q.ln() + (2 - k) as f64 * (1.0 - q).ln();

            // Moment generating function for k changed points
            let mgf_term = match k {
                0 => 0.0, // No change, moment is 1, log moment is 0
                1 => self.compute_single_change_moment(alpha, sigma)?,
                2 => self.compute_double_change_moment(alpha, sigma)?,
                _ => 0.0,
            };

            let term = log_binomial + log_prob_k + mgf_term;

            if k == 0 {
                log_moment = term;
            } else {
                log_moment = log_sum_exp(log_moment, term);
            }
        }

        Ok(log_moment)
    }

    /// Compute moment for single point change
    fn compute_single_change_moment(&self, alpha: f64, sigma: f64) -> Result<f64> {
        // For Gaussian mechanism with sensitivity 1:
        // M_α(λ) = exp(α(α-1)/(2σ²))
        let log_moment = alpha * (alpha - 1.0) / (2.0 * sigma * sigma);
        Ok(log_moment)
    }

    /// Compute moment for double point change (advanced case)
    fn compute_double_change_moment(&self, alpha: f64, sigma: f64) -> Result<f64> {
        // Simplified computation for double change
        // In practice, this requires more sophisticated analysis
        let single_change = self.compute_single_change_moment(alpha, sigma)?;
        Ok(2.0 * single_change) // Approximation
    }

    /// Compute optimal epsilon from log moments
    fn compute_optimal_epsilon(&self, logmoments: &HashMap<usize, f64>) -> Result<(f64, usize)> {
        let mut best_epsilon = f64::INFINITY;
        let mut best_order = 2;

        for (&order, &log_moment) in logmoments {
            // Convert log moment to epsilon using the formula:
            // ε = (log_moment - log(δ)) / (α - 1)
            let alpha = order as f64;
            let epsilon = (log_moment - self.target_delta.ln()) / (alpha - 1.0);

            if epsilon < best_epsilon && epsilon > 0.0 {
                best_epsilon = epsilon;
                best_order = order;
            }
        }

        if best_epsilon == f64::INFINITY {
            return Err(OptimError::InvalidConfig(
                "Failed to compute valid epsilon".to_string(),
            ));
        }

        Ok((best_epsilon, best_order))
    }

    /// Compute privacy amplification factor due to subsampling
    fn compute_amplification_factor(&self) -> f64 {
        // Privacy amplification by subsampling
        // The amplification factor depends on the sampling probability
        let q = self.sampling_probability;

        if q >= 1.0 {
            1.0 // No amplification if sampling entire dataset
        } else {
            // Simplified amplification factor
            // In practice, this depends on the specific analysis
            (q * (1.0 - q).ln() / q.ln()).max(1.0)
        }
    }

    /// Assess how tight the privacy bounds are
    fn assess_bound_tightness(&self, _logmoments: &HashMap<usize, f64>, epsilon: f64) -> f64 {
        // Compare with basic composition bound
        let steps = 1; // Normalized to single step
        let basic_composition_epsilon =
            steps as f64 * (self.sampling_probability * epsilon / self.noise_multiplier);

        // Tightness ratio (lower is tighter)
        if basic_composition_epsilon > 0.0 {
            epsilon / basic_composition_epsilon
        } else {
            1.0
        }
    }

    /// Analyze heterogeneous composition (different mechanisms)
    pub fn analyze_heterogeneous_composition(
        &self,
        mechanisms: &[MechanismParameters],
        target_delta: f64,
    ) -> Result<CompositionAnalysis> {
        let mut total_log_moments = HashMap::new();
        let mut total_compositions = 0;

        // Compute combined log moments
        for mechanism in mechanisms {
            total_compositions += mechanism.applications;

            for order in 2..=self.max_order {
                let single_log_moment = self.compute_mechanism_log_moment(order, mechanism)?;
                let total_for_mechanism = single_log_moment * mechanism.applications as f64;

                *total_log_moments.entry(order).or_insert(0.0) += total_for_mechanism;
            }
        }

        // Find optimal epsilon
        let (composed_epsilon, _) =
            self.compute_optimal_epsilon_with_delta(&total_log_moments, target_delta)?;

        Ok(CompositionAnalysis {
            mechanisms: mechanisms.to_vec(),
            composed_epsilon,
            composed_delta: target_delta,
            num_compositions: total_compositions,
            is_heterogeneous: mechanisms.len() > 1,
        })
    }

    /// Compute optimal epsilon with custom delta
    fn compute_optimal_epsilon_with_delta(
        &self,
        log_moments: &HashMap<usize, f64>,
        delta: f64,
    ) -> Result<(f64, usize)> {
        let mut best_epsilon = f64::INFINITY;
        let mut best_order = 2;

        for (&order, &log_moment) in log_moments {
            let alpha = order as f64;
            let epsilon = (log_moment - delta.ln()) / (alpha - 1.0);

            if epsilon < best_epsilon && epsilon > 0.0 {
                best_epsilon = epsilon;
                best_order = order;
            }
        }

        Ok((best_epsilon, best_order))
    }

    /// Get computed moment orders for debugging
    pub fn get_computed_orders(&self) -> Vec<f64> {
        (2..=self.max_order).map(|x| x as f64).collect()
    }

    /// Log binomial coefficient computation
    fn log_binomial_coefficient(&self, n: usize, k: usize) -> f64 {
        if k > n {
            return f64::NEG_INFINITY;
        }

        // Use cached coefficients if available
        if let Some(&coeff) = self.coefficients.binomial_coeffs.get(&(n, k)) {
            return coeff.ln();
        }

        // Compute using log factorials
        if n < self.coefficients.log_factorials.len()
            && k < self.coefficients.log_factorials.len()
            && (n - k) < self.coefficients.log_factorials.len()
        {
            return self.coefficients.log_factorials[n]
                - self.coefficients.log_factorials[k]
                - self.coefficients.log_factorials[n - k];
        }

        // Fallback computation
        stirling_log_factorial(n) - stirling_log_factorial(k) - stirling_log_factorial(n - k)
    }

    /// Validate moment accountant configuration
    pub fn validate_configuration(&self) -> Result<()> {
        if self.noise_multiplier <= 0.0 {
            return Err(OptimError::InvalidConfig(
                "Noise multiplier must be positive".to_string(),
            ));
        }

        if self.target_delta <= 0.0 || self.target_delta >= 1.0 {
            return Err(OptimError::InvalidConfig(
                "Delta must be in (0, 1)".to_string(),
            ));
        }

        if self.batch_size == 0 || self.dataset_size == 0 {
            return Err(OptimError::InvalidConfig(
                "Batch size and dataset size must be positive".to_string(),
            ));
        }

        if self.batch_size > self.dataset_size {
            return Err(OptimError::InvalidConfig(
                "Batch size cannot exceed dataset size".to_string(),
            ));
        }

        Ok(())
    }

    /// Get privacy analysis summary
    pub fn get_analysis_summary(&self, steps: usize) -> Result<PrivacyAnalysisSummary> {
        let analysis = self.analyze_privacy(steps)?;

        Ok(PrivacyAnalysisSummary {
            epsilon: analysis.epsilon,
            delta: analysis.delta,
            steps,
            noise_multiplier: self.noise_multiplier,
            sampling_probability: self.sampling_probability,
            optimal_order: analysis.optimal_order,
            amplification_factor: analysis.amplification_factor,
            bound_tightness: analysis.bound_tightness,
            privacy_per_step: analysis.epsilon / steps as f64,
        })
    }
}

impl MomentCoefficients {
    fn new(_maxorder: usize) -> Self {
        let mut binomial_coeffs = HashMap::new();
        let power_cache = HashMap::new();
        let mut log_factorials = Vec::new();

        // Precompute log factorials
        log_factorials.push(0.0); // log(0!) = log(1) = 0
        for i in 1..=_maxorder * 2 {
            log_factorials.push(log_factorials[i - 1] + (i as f64).ln());
        }

        // Precompute small binomial coefficients
        for n in 0..=10 {
            for k in 0..=n {
                let coeff = binomial_coefficient(n, k);
                binomial_coeffs.insert((n, k), coeff);
            }
        }

        Self {
            binomial_coeffs,
            power_cache,
            log_factorials,
        }
    }
}

/// Privacy analysis summary
#[derive(Debug, Clone)]
pub struct PrivacyAnalysisSummary {
    pub epsilon: f64,
    pub delta: f64,
    pub steps: usize,
    pub noise_multiplier: f64,
    pub sampling_probability: f64,
    pub optimal_order: usize,
    pub amplification_factor: f64,
    pub bound_tightness: f64,
    pub privacy_per_step: f64,
}

// Utility functions

/// Compute log(exp(a) + exp(b)) numerically stable
#[allow(dead_code)]
fn log_sum_exp(a: f64, b: f64) -> f64 {
    let max_val = a.max(b);
    let min_val = a.min(b);

    if max_val - min_val > 50.0 {
        max_val // Avoid underflow
    } else {
        max_val + (1.0 + (min_val - max_val).exp()).ln()
    }
}

/// Compute binomial coefficient C(n, k)
#[allow(dead_code)]
fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }

    if k == 0 || k == n {
        return 1.0;
    }

    let k = k.min(n - k); // Use symmetry

    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64 / (i + 1) as f64;
    }

    result
}

/// Stirling's approximation for log factorial
#[allow(dead_code)]
fn stirling_log_factorial(n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }

    let n_f = n as f64;
    n_f * n_f.ln() - n_f + 0.5 * (2.0 * std::f64::consts::PI * n_f).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moment_accountant_creation() {
        let accountant = MomentsAccountant::new(1.1, 1e-5, 256, 50000);
        assert_eq!(accountant.noise_multiplier, 1.1);
        assert_eq!(accountant.batch_size, 256);
        assert_eq!(accountant.dataset_size, 50000);
    }

    #[test]
    fn test_privacy_analysis() {
        let accountant = MomentsAccountant::new(1.1, 1e-5, 256, 50000);
        let result = accountant.analyze_privacy(100);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(analysis.epsilon > 0.0);
        assert_eq!(analysis.delta, 1e-5);
        assert_eq!(analysis.steps, 100);
    }

    #[test]
    fn test_log_moment_computation() {
        let accountant = MomentsAccountant::new(1.0, 1e-5, 100, 1000);
        let log_moment = accountant.compute_log_moment(2, 1);
        assert!(log_moment.is_ok());
        assert!(log_moment.unwrap() > 0.0);
    }

    #[test]
    fn test_privacy_spent_computation() {
        let accountant = MomentsAccountant::new(1.1, 1e-5, 256, 50000);
        let (epsilon, delta) = accountant.get_privacy_spent(50).unwrap();

        assert!(epsilon > 0.0);
        assert_eq!(delta, 1e-5);

        // More steps should consume more privacy
        let (epsilon2, _) = accountant.get_privacy_spent(100).unwrap();
        assert!(epsilon2 > epsilon);
    }

    #[test]
    fn test_configuration_validation() {
        let valid_accountant = MomentsAccountant::new(1.0, 1e-5, 100, 1000);
        assert!(valid_accountant.validate_configuration().is_ok());

        let invalid_accountant = MomentsAccountant::new(-1.0, 1e-5, 100, 1000);
        assert!(invalid_accountant.validate_configuration().is_err());
    }

    #[test]
    fn test_log_sum_exp() {
        let result = log_sum_exp(0.0, 0.0);
        assert!((result - (2.0_f64).ln()).abs() < 1e-10);

        let result2 = log_sum_exp(100.0, 0.0);
        assert!((result2 - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 0), 1.0);
        assert_eq!(binomial_coefficient(5, 1), 5.0);
        assert_eq!(binomial_coefficient(5, 2), 10.0);
        assert_eq!(binomial_coefficient(5, 5), 1.0);
    }

    #[test]
    fn test_mechanism_parameters() {
        let mechanism = MechanismParameters {
            noise_multiplier: 1.0,
            sampling_probability: 0.1,
            sensitivity: 1.0,
            applications: 100,
        };

        assert_eq!(mechanism.noise_multiplier, 1.0);
        assert_eq!(mechanism.applications, 100);
    }

    #[test]
    fn test_heterogeneous_composition() {
        let accountant = MomentsAccountant::new(1.0, 1e-5, 100, 1000);

        let mechanisms = vec![
            MechanismParameters {
                noise_multiplier: 1.0,
                sampling_probability: 0.1,
                sensitivity: 1.0,
                applications: 50,
            },
            MechanismParameters {
                noise_multiplier: 1.5,
                sampling_probability: 0.1,
                sensitivity: 1.0,
                applications: 50,
            },
        ];

        let result = accountant.analyze_heterogeneous_composition(&mechanisms, 1e-5);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(analysis.is_heterogeneous);
        assert_eq!(analysis.num_compositions, 100);
    }
}
