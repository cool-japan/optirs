use std::fmt::Debug;
// Regularization strategies specific to transformer optimization
//
// This module implements transformer-specific regularization techniques that
// work in conjunction with the attention mechanisms and optimization dynamics.

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

use crate::error::{OptimError, Result};

/// Regularization strategies for transformer optimization
#[derive(Debug, Clone, Copy)]
pub enum RegularizationStrategy {
    /// No regularization
    None,
    /// Standard L2 weight decay
    L2WeightDecay,
    /// L1 sparsity regularization
    L1Sparsity,
    /// Attention entropy regularization
    AttentionEntropy,
    /// Gradient penalty regularization
    GradientPenalty,
    /// Spectral normalization
    SpectralNorm,
    /// Attention diversity regularization
    AttentionDiversity,
    /// Parameter orthogonality constraints
    Orthogonality,
    /// Adaptive regularization based on training dynamics
    Adaptive,
}

/// Transformer regularizer
#[derive(Debug, Clone)]
pub struct TransformerRegularizer<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Regularization strategy
    strategy: RegularizationStrategy,

    /// Regularization parameters
    regularization_params: RegularizationParams<T>,

    /// Attention pattern history for entropy calculations
    attention_history: Vec<Array3<T>>,

    /// Parameter statistics for adaptive regularization
    parameter_stats: HashMap<String, ParameterStatistics<T>>,

    /// Spectral normalization state
    spectral_state: Option<SpectralNormState<T>>,

    /// Step counter for adaptive strategies
    step_count: usize,
}

/// Regularization parameters
#[derive(Debug, Clone)]
pub struct RegularizationParams<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// L2 regularization coefficient
    pub l2_weight: T,

    /// L1 regularization coefficient
    pub l1_weight: T,

    /// Attention entropy regularization weight
    pub entropy_weight: T,

    /// Gradient penalty coefficient
    pub gradient_penalty_weight: T,

    /// Spectral normalization power iterations
    pub spectral_iterations: usize,

    /// Attention diversity weight
    pub diversity_weight: T,

    /// Orthogonality constraint weight
    pub orthogonality_weight: T,

    /// Adaptive regularization base strength
    pub adaptive_base_strength: T,

    /// Adaptive regularization decay rate, applied once per `adaptive_decay_steps`
    pub adaptive_decay_rate: T,

    /// Number of steps over which `adaptive_decay_rate` is applied once
    pub adaptive_decay_steps: usize,

    /// Lower bound on the adaptive regularization strength
    pub adaptive_min_strength: T,
}

/// Parameter statistics for adaptive regularization
#[derive(Debug, Clone)]
pub struct ParameterStatistics<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Running mean of parameter magnitudes
    mean_magnitude: T,

    /// Running variance of parameter magnitudes
    var_magnitude: T,

    /// Gradient-to-parameter ratio
    grad_param_ratio: T,

    /// Update count
    update_count: usize,
}

/// State for spectral normalization
#[derive(Debug, Clone)]
pub struct SpectralNormState<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Left singular vectors for each weight matrix
    u_vectors: HashMap<String, Array1<T>>,

    /// Right singular vectors for each weight matrix
    v_vectors: HashMap<String, Array1<T>>,

    /// Spectral norms for each weight matrix
    spectral_norms: HashMap<String, T>,
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync
            + 'static,
    > TransformerRegularizer<T>
{
    /// Create new transformer regularizer
    pub fn new(strategy: RegularizationStrategy) -> Self {
        Self {
            strategy,
            regularization_params: RegularizationParams::default(),
            attention_history: Vec::new(),
            parameter_stats: HashMap::new(),
            spectral_state: None,
            step_count: 0,
        }
    }

    /// Create with custom parameters
    pub fn new_with_params(
        strategy: RegularizationStrategy,
        params: RegularizationParams<T>,
    ) -> Self {
        Self {
            strategy,
            regularization_params: params,
            attention_history: Vec::new(),
            parameter_stats: HashMap::new(),
            spectral_state: None,
            step_count: 0,
        }
    }

    /// Apply regularization to parameters and gradients.
    ///
    /// `parameters` is taken by mutable reference because the spectral
    /// normalization strategy rescales the weight matrices themselves.
    pub fn apply_regularization(
        &mut self,
        parameters: &mut HashMap<String, Array2<T>>,
        gradients: &mut HashMap<String, Array2<T>>,
        attention_patterns: Option<&Array3<T>>,
    ) -> Result<T> {
        self.step_count += 1;

        // Reject shape mismatches instead of panicking inside ndarray ops.
        for (name, param) in parameters.iter() {
            if let Some(grad) = gradients.get(name) {
                if param.dim() != grad.dim() {
                    return Err(OptimError::InvalidConfig(format!(
                        "Parameter '{}' has shape {:?} but its gradient has shape {:?}",
                        name,
                        param.dim(),
                        grad.dim()
                    )));
                }
            }
        }

        // Store attention patterns for entropy regularization
        if let Some(attention) = attention_patterns {
            self.attention_history.push(attention.clone());
            // Keep only recent history
            if self.attention_history.len() > 100 {
                self.attention_history.remove(0);
            }
        }

        match self.strategy {
            RegularizationStrategy::None => Ok(T::zero()),
            RegularizationStrategy::L2WeightDecay => {
                self.apply_l2_regularization(&*parameters, gradients)
            }
            RegularizationStrategy::L1Sparsity => {
                self.apply_l1_regularization(&*parameters, gradients)
            }
            RegularizationStrategy::AttentionEntropy => {
                self.apply_attention_entropy_regularization(gradients, attention_patterns)
            }
            RegularizationStrategy::GradientPenalty => self.apply_gradient_penalty(gradients),
            RegularizationStrategy::SpectralNorm => self.apply_spectral_normalization(parameters),
            RegularizationStrategy::AttentionDiversity => {
                self.apply_attention_diversity_regularization(gradients, attention_patterns)
            }
            RegularizationStrategy::Orthogonality => {
                self.apply_orthogonality_regularization(&*parameters, gradients)
            }
            RegularizationStrategy::Adaptive => {
                self.apply_adaptive_regularization(&*parameters, gradients)
            }
        }
    }

    /// Apply L2 weight decay regularization
    fn apply_l2_regularization(
        &self,
        parameters: &HashMap<String, Array2<T>>,
        gradients: &mut HashMap<String, Array2<T>>,
    ) -> Result<T> {
        let mut total_reg_loss = T::zero();

        for (param_name, param_values) in parameters {
            if let Some(grad) = gradients.get_mut(param_name) {
                // Add L2 penalty to gradients
                let l2_grad = param_values * self.regularization_params.l2_weight;
                *grad = grad.clone() + &l2_grad;

                // Compute L2 loss contribution
                let l2_loss = param_values
                    .iter()
                    .map(|&x| x * x)
                    .fold(T::zero(), |a, b| a + b);
                total_reg_loss = total_reg_loss
                    + l2_loss
                        * self.regularization_params.l2_weight
                        * scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero());
            }
        }

        Ok(total_reg_loss)
    }

    /// Apply L1 sparsity regularization
    fn apply_l1_regularization(
        &self,
        parameters: &HashMap<String, Array2<T>>,
        gradients: &mut HashMap<String, Array2<T>>,
    ) -> Result<T> {
        let mut total_reg_loss = T::zero();

        for (param_name, param_values) in parameters {
            if let Some(grad) = gradients.get_mut(param_name) {
                // Add L1 subgradient to gradients. At exactly zero the
                // subdifferential contains zero, so no force is applied - the
                // previous `-lambda` branch actively pushed weights away from 0.
                let l1_grad = param_values.mapv(|x| {
                    if x > T::zero() {
                        self.regularization_params.l1_weight
                    } else if x < T::zero() {
                        -self.regularization_params.l1_weight
                    } else {
                        T::zero()
                    }
                });
                *grad = grad.clone() + &l1_grad;

                // Compute L1 loss contribution
                let l1_loss = param_values
                    .iter()
                    .map(|&x| x.abs())
                    .fold(T::zero(), |a, b| a + b);
                total_reg_loss = total_reg_loss + l1_loss * self.regularization_params.l1_weight;
            }
        }

        Ok(total_reg_loss)
    }

    /// Apply attention entropy regularization
    fn apply_attention_entropy_regularization(
        &self,
        gradients: &mut HashMap<String, Array2<T>>,
        attention_patterns: Option<&Array3<T>>,
    ) -> Result<T> {
        if let Some(attention) = attention_patterns {
            let entropy_penalty = self.compute_attention_entropy(attention)?;

            // Apply entropy penalty to attention-related gradients
            if let Some(attention_grad) = gradients.get_mut("attention_weights") {
                let entropy_grad = self.compute_entropy_gradient(attention)?;
                *attention_grad = attention_grad.clone() + &entropy_grad;
            }

            Ok(entropy_penalty * self.regularization_params.entropy_weight)
        } else {
            Ok(T::zero())
        }
    }

    /// Apply gradient penalty regularization.
    ///
    /// The penalty is `lambda * ||g||^2`; descending on it shrinks the gradient
    /// by `g <- g * (1 - 2*lambda)`. The previous implementation *added*
    /// `2*lambda*g`, which amplified exactly the gradients it claimed to
    /// penalize. The damping factor is clamped to `[0, 1]` so a large lambda
    /// cannot flip the descent direction.
    fn apply_gradient_penalty(&self, gradients: &mut HashMap<String, Array2<T>>) -> Result<T> {
        let mut total_penalty = T::zero();
        let two: T = scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::one());
        let damping = (T::one() - two * self.regularization_params.gradient_penalty_weight)
            .max(T::zero())
            .min(T::one());

        for grad in gradients.values_mut() {
            let grad_norm_squared = grad.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b);
            *grad = grad.clone() * damping;

            total_penalty = total_penalty
                + grad_norm_squared * self.regularization_params.gradient_penalty_weight;
        }

        Ok(total_penalty)
    }

    /// Apply spectral normalization.
    ///
    /// Spectral normalization constrains the *weights*, dividing each weight
    /// matrix by its largest singular value so that its spectral norm becomes
    /// one. Scaling the gradients instead (as the previous implementation did)
    /// leaves the weights unconstrained and merely changes the step size.
    fn apply_spectral_normalization(
        &mut self,
        parameters: &mut HashMap<String, Array2<T>>,
    ) -> Result<T> {
        let spectral_state = self
            .spectral_state
            .get_or_insert_with(|| SpectralNormState {
                u_vectors: HashMap::new(),
                v_vectors: HashMap::new(),
                spectral_norms: HashMap::new(),
            });

        let mut total_reg_loss = T::zero();
        let spectral_iterations = self.regularization_params.spectral_iterations.max(1);

        for (param_name, param_values) in parameters.iter_mut() {
            let spectral_norm = Self::compute_spectral_norm_static(
                param_name,
                param_values,
                spectral_state,
                spectral_iterations,
            )?;

            if spectral_norm > T::one() {
                // Rescale the weight matrix itself.
                *param_values = param_values.clone() / spectral_norm;
                spectral_state
                    .spectral_norms
                    .insert(param_name.to_string(), T::one());

                total_reg_loss = total_reg_loss + (spectral_norm - T::one()).powi(2);
            }
        }

        Ok(total_reg_loss)
    }

    /// Apply attention diversity regularization
    fn apply_attention_diversity_regularization(
        &self,
        gradients: &mut HashMap<String, Array2<T>>,
        attention_patterns: Option<&Array3<T>>,
    ) -> Result<T> {
        if let Some(attention) = attention_patterns {
            let diversity_penalty = self.compute_attention_diversity(attention)?;

            // Apply diversity penalty to attention gradients
            if let Some(attention_grad) = gradients.get_mut("attention_weights") {
                let diversity_grad = self.compute_diversity_gradient(attention)?;
                *attention_grad = attention_grad.clone() + &diversity_grad;
            }

            Ok(diversity_penalty * self.regularization_params.diversity_weight)
        } else {
            Ok(T::zero())
        }
    }

    /// Apply orthogonality regularization
    fn apply_orthogonality_regularization(
        &self,
        parameters: &HashMap<String, Array2<T>>,
        gradients: &mut HashMap<String, Array2<T>>,
    ) -> Result<T> {
        let mut total_reg_loss = T::zero();

        for (param_name, param_values) in parameters {
            // Apply orthogonality constraint to weight matrices
            if param_name.contains("weight") && param_values.nrows() == param_values.ncols() {
                let orthogonality_penalty = self.compute_orthogonality_penalty(param_values)?;

                if let Some(grad) = gradients.get_mut(param_name) {
                    let ortho_grad = self.compute_orthogonality_gradient(param_values)?;
                    *grad = grad.clone() + &ortho_grad;
                }

                total_reg_loss = total_reg_loss
                    + orthogonality_penalty * self.regularization_params.orthogonality_weight;
            }
        }

        Ok(total_reg_loss)
    }

    /// Apply adaptive regularization based on training dynamics
    fn apply_adaptive_regularization(
        &mut self,
        parameters: &HashMap<String, Array2<T>>,
        gradients: &mut HashMap<String, Array2<T>>,
    ) -> Result<T> {
        let mut total_reg_loss = T::zero();

        for (param_name, param_values) in parameters {
            // Update parameter statistics
            self.update_parameter_statistics(param_name, param_values, gradients.get(param_name));

            // Compute adaptive regularization strength
            let adaptive_strength = self.compute_adaptive_strength(param_name)?;

            if let Some(grad) = gradients.get_mut(param_name) {
                // Apply adaptive L2 regularization
                let adaptive_grad = param_values * adaptive_strength;
                *grad = grad.clone() + &adaptive_grad;

                // Compute regularization loss
                let reg_loss = param_values
                    .iter()
                    .map(|&x| x * x)
                    .fold(T::zero(), |a, b| a + b);
                total_reg_loss = total_reg_loss
                    + reg_loss
                        * adaptive_strength
                        * scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero());
            }
        }

        Ok(total_reg_loss)
    }

    /// Compute attention entropy
    fn compute_attention_entropy(&self, attention: &Array3<T>) -> Result<T> {
        let (num_heads, seq_len, _) = attention.dim();
        let mut total_entropy = T::zero();

        for h in 0..num_heads {
            for i in 0..seq_len {
                let mut entropy = T::zero();
                for j in 0..seq_len {
                    let p = attention[[h, i, j]];
                    if p > T::zero() {
                        entropy = entropy - p * p.ln();
                    }
                }
                total_entropy = total_entropy + entropy;
            }
        }

        let denominator: T = scirs2_core::numeric::NumCast::from((num_heads * seq_len) as f64)
            .unwrap_or_else(|| T::one());
        if denominator == T::zero() {
            return Ok(T::zero());
        }
        Ok(total_entropy / denominator)
    }

    /// Compute entropy gradient (simplified)
    fn compute_entropy_gradient(&self, attention: &Array3<T>) -> Result<Array2<T>> {
        let (num_heads, seq_len) = (attention.shape()[0], attention.shape()[1]);
        let mut grad = Array2::zeros((num_heads, seq_len));

        for h in 0..num_heads {
            for i in 0..seq_len {
                // Simplified entropy gradient
                grad[[h, i]] = -self.regularization_params.entropy_weight;
            }
        }

        Ok(grad)
    }

    /// Compute attention diversity penalty
    fn compute_attention_diversity(&self, attention: &Array3<T>) -> Result<T> {
        let (num_heads, seq_len, _) = attention.dim();
        let mut diversity_penalty = T::zero();

        // Penalize similarity between attention heads
        for h1 in 0..num_heads {
            for h2 in (h1 + 1)..num_heads {
                let mut similarity = T::zero();
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        similarity = similarity + attention[[h1, i, j]] * attention[[h2, i, j]];
                    }
                }
                diversity_penalty = diversity_penalty + similarity * similarity;
            }
        }

        Ok(diversity_penalty)
    }

    /// Compute diversity gradient (simplified)
    fn compute_diversity_gradient(&self, attention: &Array3<T>) -> Result<Array2<T>> {
        let (num_heads, seq_len) = (attention.shape()[0], attention.shape()[1]);
        let mut grad = Array2::zeros((num_heads, seq_len));

        // Simplified diversity gradient computation
        for h in 0..num_heads {
            for i in 0..seq_len {
                grad[[h, i]] = scirs2_core::numeric::NumCast::from(0.1)
                    .unwrap_or_else(|| T::zero())
                    * self.regularization_params.diversity_weight;
            }
        }

        Ok(grad)
    }

    /// Compute `W^T W - I` once for reuse by the penalty and its gradient.
    fn gram_minus_identity(matrix: &Array2<T>) -> Array2<T> {
        let mut gram = matrix.t().dot(matrix);
        let n = gram.nrows().min(gram.ncols());
        for i in 0..n {
            gram[[i, i]] = gram[[i, i]] - T::one();
        }
        gram
    }

    /// Compute orthogonality penalty for square matrices
    fn compute_orthogonality_penalty(&self, matrix: &Array2<T>) -> Result<T> {
        if matrix.nrows() != matrix.ncols() {
            return Ok(T::zero());
        }

        // ||W^T W - I||_F^2
        let residual = Self::gram_minus_identity(matrix);
        Ok(residual
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |a, b| a + b))
    }

    /// Compute the gradient of `||W^T W - I||_F^2` with respect to `W`.
    ///
    /// The closed form is `4 * W * (W^T W - I)`, i.e. two matrix products
    /// instead of the quartic loop nest the previous version used.
    fn compute_orthogonality_gradient(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        if matrix.nrows() != matrix.ncols() {
            return Ok(Array2::zeros(matrix.dim()));
        }

        let residual = Self::gram_minus_identity(matrix);
        let four: T = scirs2_core::numeric::NumCast::from(4.0).unwrap_or_else(|| T::one());
        let grad = matrix.dot(&residual) * (four * self.regularization_params.orthogonality_weight);

        Ok(grad)
    }

    /// Update parameter statistics for adaptive regularization
    fn update_parameter_statistics(
        &mut self,
        param_name: &str,
        parameters: &Array2<T>,
        gradients: Option<&Array2<T>>,
    ) {
        let param_magnitude = parameters
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |a, b| a + b)
            .sqrt();

        let stats = self
            .parameter_stats
            .entry(param_name.to_string())
            .or_insert(ParameterStatistics::new());

        stats.update_count += 1;
        let alpha = scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero());

        // Update running mean and the matching EWMA variance. The variance uses
        // the *pre-update* mean, which is the standard incremental exponentially
        // weighted form `v <- (1-a)(v + a·d²)` with `d = x - mean_prev`; it used
        // to be initialized to zero and never touched, so
        // `compute_adaptive_strength` had no measure of how volatile a
        // parameter's magnitude was.
        let deviation = param_magnitude - stats.mean_magnitude;
        stats.var_magnitude =
            (T::one() - alpha) * (stats.var_magnitude + alpha * deviation * deviation);
        stats.mean_magnitude = stats.mean_magnitude * (T::one() - alpha) + param_magnitude * alpha;

        // Update gradient-to-parameter ratio if gradients available
        if let Some(grad) = gradients {
            let grad_magnitude = grad
                .iter()
                .map(|&x| x * x)
                .fold(T::zero(), |a, b| a + b)
                .sqrt();
            if param_magnitude > T::zero() {
                let ratio = grad_magnitude / param_magnitude;
                stats.grad_param_ratio =
                    stats.grad_param_ratio * (T::one() - alpha) + ratio * alpha;
            }
        }
    }

    /// Compute adaptive regularization strength.
    ///
    /// The decay is applied on a *schedule scale* (`decay_rate ^ (step /
    /// decay_steps)`) and floored at `adaptive_min_strength`. Applying the
    /// decay once per step, as the previous implementation did, drove the
    /// strength to ~1e-9 by step 2000 with the default `0.99` rate, silently
    /// disabling the regularizer.
    fn compute_adaptive_strength(&self, param_name: &str) -> Result<T> {
        let base_strength = self.regularization_params.adaptive_base_strength;
        let decay_steps = self.regularization_params.adaptive_decay_steps.max(1) as f64;
        let progress: T = scirs2_core::numeric::NumCast::from(self.step_count as f64 / decay_steps)
            .unwrap_or_else(|| T::zero());
        let step_decay = self
            .regularization_params
            .adaptive_decay_rate
            .powf(progress);

        // Scale down with the parameter's magnitude, and *up* with how volatile
        // that magnitude has been: a parameter whose norm swings around needs
        // more regularization than a settled one of the same size.
        let magnitude_factor = match self.parameter_stats.get(param_name) {
            Some(stats) => {
                let volatility = stats.var_magnitude.sqrt();
                (T::one() + volatility) / (T::one() + stats.mean_magnitude)
            }
            None => T::one(),
        };

        Ok((base_strength * magnitude_factor * step_decay)
            .max(self.regularization_params.adaptive_min_strength))
    }

    /// Compute spectral norm using power iteration (static version)
    fn compute_spectral_norm_static(
        param_name: &str,
        matrix: &Array2<T>,
        spectral_state: &mut SpectralNormState<T>,
        spectral_iterations: usize,
    ) -> Result<T> {
        let (m, n) = matrix.dim();
        if m == 0 || n == 0 {
            return Ok(T::zero());
        }

        // Initialize (or re-initialize on a dimension change) the cached
        // singular vectors. Reusing a stale vector of the wrong length would
        // index out of bounds during the power iteration.
        let u_stale = spectral_state
            .u_vectors
            .get(param_name)
            .is_none_or(|u| u.len() != m);
        let v_stale = spectral_state
            .v_vectors
            .get(param_name)
            .is_none_or(|v| v.len() != n);
        if u_stale || v_stale {
            let scale_u: T = scirs2_core::numeric::NumCast::from(1.0 / (m as f64).sqrt())
                .unwrap_or_else(|| T::one());
            let scale_v: T = scirs2_core::numeric::NumCast::from(1.0 / (n as f64).sqrt())
                .unwrap_or_else(|| T::one());
            spectral_state
                .u_vectors
                .insert(param_name.to_string(), Array1::from_elem(m, scale_u));
            spectral_state
                .v_vectors
                .insert(param_name.to_string(), Array1::from_elem(n, scale_v));
        }

        // Power iteration to find largest singular value
        let mut u = spectral_state.u_vectors[param_name].clone();
        let mut v = spectral_state.v_vectors[param_name].clone();

        for _ in 0..spectral_iterations {
            // v = W^T u / ||W^T u||
            let mut new_v = Array1::zeros(n);
            for j in 0..n {
                for i in 0..m {
                    new_v[j] = new_v[j] + matrix[[i, j]] * u[i];
                }
            }
            let v_norm = new_v
                .iter()
                .map(|&x| x * x)
                .fold(T::zero(), |a, b| a + b)
                .sqrt();
            if v_norm > T::zero() {
                v = new_v / v_norm;
            }

            // u = W v / ||W v||
            let mut new_u = Array1::zeros(m);
            for i in 0..m {
                for j in 0..n {
                    new_u[i] = new_u[i] + matrix[[i, j]] * v[j];
                }
            }
            let u_norm = new_u
                .iter()
                .map(|&x| x * x)
                .fold(T::zero(), |a, b| a + b)
                .sqrt();
            if u_norm > T::zero() {
                u = new_u / u_norm;
            }
        }

        // Compute spectral norm: u^T W v
        let mut spectral_norm = T::zero();
        for i in 0..m {
            for j in 0..n {
                spectral_norm = spectral_norm + u[i] * matrix[[i, j]] * v[j];
            }
        }

        // Update state
        spectral_state.u_vectors.insert(param_name.to_string(), u);
        spectral_state.v_vectors.insert(param_name.to_string(), v);
        spectral_state
            .spectral_norms
            .insert(param_name.to_string(), spectral_norm);

        Ok(spectral_norm)
    }

    /// Get regularization statistics
    pub fn get_statistics(&self) -> HashMap<String, T> {
        let mut stats = HashMap::new();

        stats.insert(
            "step_count".to_string(),
            scirs2_core::numeric::NumCast::from(self.step_count as f64)
                .unwrap_or_else(|| T::zero()),
        );
        stats.insert(
            "attention_history_length".to_string(),
            scirs2_core::numeric::NumCast::from(self.attention_history.len() as f64)
                .unwrap_or_else(|| T::zero()),
        );

        if let Some(ref spectral_state) = self.spectral_state {
            for (param_name, &norm) in &spectral_state.spectral_norms {
                stats.insert(format!("spectral_norm_{}", param_name), norm);
            }
        }

        stats
    }

    /// Reset regularizer state
    pub fn reset(&mut self) {
        self.attention_history.clear();
        self.parameter_stats.clear();
        self.spectral_state = None;
        self.step_count = 0;
    }

    /// Update strategy
    pub fn set_strategy(&mut self, strategy: RegularizationStrategy) {
        self.strategy = strategy;
    }

    /// Update parameters
    pub fn set_parameters(&mut self, params: RegularizationParams<T>) {
        self.regularization_params = params;
    }
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync
            + 'static,
    > ParameterStatistics<T>
{
    fn new() -> Self {
        Self {
            mean_magnitude: T::zero(),
            var_magnitude: T::zero(),
            grad_param_ratio: T::zero(),
            update_count: 0,
        }
    }
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync
            + 'static,
    > Default for RegularizationParams<T>
{
    fn default() -> Self {
        Self {
            l2_weight: scirs2_core::numeric::NumCast::from(0.01).unwrap_or_else(|| T::zero()),
            l1_weight: scirs2_core::numeric::NumCast::from(0.001).unwrap_or_else(|| T::zero()),
            entropy_weight: scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero()),
            gradient_penalty_weight: scirs2_core::numeric::NumCast::from(0.01)
                .unwrap_or_else(|| T::zero()),
            spectral_iterations: 8,
            diversity_weight: scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero()),
            orthogonality_weight: scirs2_core::numeric::NumCast::from(0.01)
                .unwrap_or_else(|| T::zero()),
            adaptive_base_strength: scirs2_core::numeric::NumCast::from(0.01)
                .unwrap_or_else(|| T::zero()),
            adaptive_decay_rate: scirs2_core::numeric::NumCast::from(0.99)
                .unwrap_or_else(|| T::zero()),
            adaptive_decay_steps: 1000,
            adaptive_min_strength: scirs2_core::numeric::NumCast::from(1e-4)
                .unwrap_or_else(|| T::zero()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(values: &[f64], rows: usize, cols: usize) -> HashMap<String, Array2<f64>> {
        let mut map = HashMap::new();
        map.insert(
            "weight".to_string(),
            Array2::from_shape_vec((rows, cols), values.to_vec()).expect("valid shape"),
        );
        map
    }

    #[test]
    fn gradient_penalty_shrinks_gradients() {
        let mut regularizer =
            TransformerRegularizer::<f64>::new(RegularizationStrategy::GradientPenalty);
        let mut parameters = params(&[1.0, 1.0, 1.0, 1.0], 2, 2);
        let mut gradients = params(&[2.0, -2.0, 4.0, -4.0], 2, 2);

        let _ = regularizer
            .apply_regularization(&mut parameters, &mut gradients, None)
            .expect("regularization must succeed");

        let grad = gradients.get("weight").expect("gradient present");
        assert!(grad[[0, 0]].abs() < 2.0, "penalty amplified: {grad:?}");
        assert!(grad[[0, 0]] > 0.0, "penalty flipped the descent direction");
    }

    #[test]
    fn shape_mismatch_is_an_error_not_a_panic() {
        let mut regularizer =
            TransformerRegularizer::<f64>::new(RegularizationStrategy::L2WeightDecay);
        let mut parameters = params(&[1.0, 1.0, 1.0, 1.0], 2, 2);
        let mut gradients = params(&[1.0, 1.0], 1, 2);

        let result = regularizer.apply_regularization(&mut parameters, &mut gradients, None);
        assert!(result.is_err(), "shape mismatch should be reported");
    }

    #[test]
    fn l1_subgradient_is_zero_at_zero() {
        let mut regularizer =
            TransformerRegularizer::<f64>::new(RegularizationStrategy::L1Sparsity);
        let mut parameters = params(&[0.0, 1.0, -1.0, 0.0], 2, 2);
        let mut gradients = params(&[0.0, 0.0, 0.0, 0.0], 2, 2);

        let _ = regularizer
            .apply_regularization(&mut parameters, &mut gradients, None)
            .expect("regularization must succeed");

        let grad = gradients.get("weight").expect("gradient present");
        assert_eq!(grad[[0, 0]], 0.0, "zero weights must not be pushed away");
        assert!(grad[[0, 1]] > 0.0);
        assert!(grad[[1, 0]] < 0.0);
    }

    #[test]
    fn spectral_normalization_rescales_weights() {
        let mut regularizer =
            TransformerRegularizer::<f64>::new(RegularizationStrategy::SpectralNorm);
        // Diagonal matrix with largest singular value 5.
        let mut parameters = params(&[5.0, 0.0, 0.0, 1.0], 2, 2);
        let mut gradients = params(&[1.0, 1.0, 1.0, 1.0], 2, 2);

        let _ = regularizer
            .apply_regularization(&mut parameters, &mut gradients, None)
            .expect("regularization must succeed");

        let weight = parameters.get("weight").expect("weight present");
        assert!(
            (weight[[0, 0]] - 1.0).abs() < 1e-6,
            "weights were not rescaled: {weight:?}"
        );
        // Gradients are untouched by spectral normalization.
        assert_eq!(gradients["weight"][[0, 0]], 1.0);
    }

    #[test]
    fn adaptive_strength_does_not_vanish() {
        let mut regularizer = TransformerRegularizer::<f64>::new(RegularizationStrategy::Adaptive);
        let mut parameters = params(&[0.5, 0.5, 0.5, 0.5], 2, 2);

        let mut last_strength = 0.0;
        for _ in 0..2000 {
            let mut gradients = params(&[0.1, 0.1, 0.1, 0.1], 2, 2);
            let _ = regularizer
                .apply_regularization(&mut parameters, &mut gradients, None)
                .expect("regularization must succeed");
            last_strength = regularizer
                .compute_adaptive_strength("weight")
                .expect("strength");
        }
        assert!(
            last_strength >= 1e-4,
            "adaptive strength decayed to {last_strength}"
        );
    }

    #[test]
    fn orthogonality_gradient_vanishes_for_orthogonal_matrices() {
        let regularizer = TransformerRegularizer::<f64>::new(RegularizationStrategy::Orthogonality);
        let identity = Array2::<f64>::eye(4);
        let penalty = regularizer
            .compute_orthogonality_penalty(&identity)
            .expect("penalty");
        assert!(penalty.abs() < 1e-12, "penalty {penalty} should be zero");

        let grad = regularizer
            .compute_orthogonality_gradient(&identity)
            .expect("gradient");
        assert!(grad.iter().all(|&x| x.abs() < 1e-12), "grad {grad:?}");
    }

    #[test]
    fn orthogonality_gradient_matches_closed_form() {
        let regularizer = TransformerRegularizer::<f64>::new(RegularizationStrategy::Orthogonality);
        let params_cfg = RegularizationParams::<f64> {
            orthogonality_weight: 1.0,
            ..RegularizationParams::default()
        };
        let mut regularizer = regularizer;
        regularizer.set_parameters(params_cfg);

        // W = 2I => W^T W - I = 3I => grad = 4 * W * 3I = 24 I
        let matrix = Array2::<f64>::eye(3) * 2.0;
        let grad = regularizer
            .compute_orthogonality_gradient(&matrix)
            .expect("gradient");
        assert!((grad[[0, 0]] - 24.0).abs() < 1e-9, "grad {grad:?}");
        assert!(grad[[0, 1]].abs() < 1e-12);
    }
}
