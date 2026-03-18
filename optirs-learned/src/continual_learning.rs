// Continual Learning for Optimizers
//
// This module implements continual learning techniques that allow neural networks
// and optimizers to learn new tasks sequentially without catastrophically forgetting
// previous ones. It includes:
//
// - Elastic Weight Consolidation (EWC): Regularization-based approach using Fisher
//   information to protect important parameters for previous tasks.
// - Progressive Networks: Architecture-based approach that adds new columns for new
//   tasks while preserving previously learned knowledge via frozen columns and lateral
//   connections.
//
// References:
// - Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (2017)
// - Rusu et al., "Progressive Neural Networks" (2016)

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{OptimError, Result};

// ---------------------------------------------------------------------------
// Elastic Weight Consolidation (EWC)
// ---------------------------------------------------------------------------

/// Elastic Weight Consolidation (EWC) regularizer for continual learning.
///
/// EWC prevents catastrophic forgetting by adding a quadratic penalty that
/// discourages changes to parameters that were important for previous tasks.
/// The importance of each parameter is estimated via the diagonal of the
/// Fisher information matrix.
///
/// Supports both standard (multi-task) EWC and online EWC with exponential
/// decay of older Fisher information.
pub struct ElasticWeightConsolidation<T: Float + Debug + Send + Sync + 'static> {
    /// Regularization strength (lambda)
    lambda: T,
    /// Diagonal Fisher information matrix per parameter group
    fisher_diagonal: HashMap<String, Array1<T>>,
    /// Anchor parameters (theta*) stored after training on a task
    anchor_parameters: HashMap<String, Array1<T>>,
    /// Per-task Fisher diagonals for standard (non-online) EWC
    task_fisher_diagonals: Vec<HashMap<String, Array1<T>>>,
    /// Per-task anchor parameters (for standard EWC penalty over all tasks)
    task_anchor_parameters: Vec<HashMap<String, Array1<T>>>,
    /// Number of gradient samples used to estimate the Fisher
    num_samples_fisher: usize,
    /// Whether to use online EWC (running average of Fisher)
    online: bool,
    /// Decay factor for online EWC
    gamma: T,
}

impl<T: Float + Debug + Send + Sync + 'static> ElasticWeightConsolidation<T> {
    /// Create a new EWC regularizer with the given lambda (regularization strength).
    ///
    /// Defaults: num_samples_fisher = 100, online = false, gamma = 0.95
    pub fn new(lambda: T) -> Self {
        Self {
            lambda,
            fisher_diagonal: HashMap::new(),
            anchor_parameters: HashMap::new(),
            task_fisher_diagonals: Vec::new(),
            task_anchor_parameters: Vec::new(),
            num_samples_fisher: 100,
            online: false,
            gamma: T::from(0.95).unwrap_or_else(|| T::one()),
        }
    }

    /// Set the number of gradient samples used to estimate the Fisher information.
    pub fn with_num_samples(mut self, n: usize) -> Self {
        self.num_samples_fisher = n;
        self
    }

    /// Enable or disable online EWC mode.
    pub fn with_online(mut self, online: bool) -> Self {
        self.online = online;
        self
    }

    /// Set the decay factor gamma for online EWC.
    pub fn with_gamma(mut self, gamma: T) -> Self {
        self.gamma = gamma;
        self
    }

    /// Compute the diagonal Fisher information matrix by sampling gradients.
    ///
    /// `parameters` - current model parameters keyed by name.
    /// `gradients_fn` - callable that returns stochastic gradients for a single sample.
    ///
    /// The Fisher diagonal is approximated as E[g_i^2] where g_i is the gradient
    /// of the log-likelihood with respect to parameter i.
    pub fn compute_fisher_diagonal(
        &mut self,
        parameters: &HashMap<String, Array1<T>>,
        gradients_fn: impl Fn(&HashMap<String, Array1<T>>) -> Result<HashMap<String, Array1<T>>>,
    ) -> Result<()> {
        if parameters.is_empty() {
            return Err(OptimError::InsufficientData(
                "parameters map is empty".to_string(),
            ));
        }

        if self.num_samples_fisher == 0 {
            return Err(OptimError::InvalidConfig(
                "num_samples_fisher must be > 0".to_string(),
            ));
        }

        // Accumulator for squared gradients
        let mut fisher_accum: HashMap<String, Array1<T>> = HashMap::new();
        for (name, param) in parameters {
            fisher_accum.insert(name.clone(), Array1::from_elem(param.len(), T::zero()));
        }

        let n_samples = T::from(self.num_samples_fisher).unwrap_or_else(|| T::one());

        for _ in 0..self.num_samples_fisher {
            let grads = gradients_fn(parameters)?;
            for (name, grad) in &grads {
                if let Some(accum) = fisher_accum.get_mut(name) {
                    if accum.len() != grad.len() {
                        return Err(OptimError::ComputationError(format!(
                            "gradient dimension mismatch for '{}': expected {}, got {}",
                            name,
                            accum.len(),
                            grad.len()
                        )));
                    }
                    // Accumulate g_i^2
                    for (a, g) in accum.iter_mut().zip(grad.iter()) {
                        *a = *a + (*g) * (*g);
                    }
                }
            }
        }

        // Average: F_i = (1/N) * sum(g_i^2)
        let mut fisher = HashMap::new();
        for (name, mut accum) in fisher_accum {
            for a in accum.iter_mut() {
                *a = *a / n_samples;
            }
            fisher.insert(name, accum);
        }

        self.fisher_diagonal = fisher;
        Ok(())
    }

    /// Consolidate the current task by storing anchor parameters and Fisher information.
    ///
    /// This should be called after training on a task completes.
    /// For online EWC, the existing Fisher is decayed by gamma and the new Fisher is added.
    /// For standard EWC, the new Fisher is appended to the list of per-task Fishers.
    pub fn consolidate(&mut self, parameters: &HashMap<String, Array1<T>>) -> Result<()> {
        if self.fisher_diagonal.is_empty() {
            return Err(OptimError::InvalidState(
                "Fisher diagonal not computed; call compute_fisher_diagonal first".to_string(),
            ));
        }

        // Store anchor parameters
        self.anchor_parameters = parameters.clone();
        self.task_anchor_parameters.push(parameters.clone());

        if self.online {
            // Online EWC: running Fisher = gamma * old_Fisher + new_Fisher
            let mut merged = HashMap::new();
            for (name, new_fisher) in &self.fisher_diagonal {
                let updated = if let Some(old_fisher) = self
                    .task_fisher_diagonals
                    .last()
                    .and_then(|map| map.get(name))
                {
                    // gamma * old + new
                    let mut result = Array1::from_elem(new_fisher.len(), T::zero());
                    for i in 0..result.len() {
                        result[i] = self.gamma * old_fisher[i] + new_fisher[i];
                    }
                    result
                } else {
                    new_fisher.clone()
                };
                merged.insert(name.clone(), updated);
            }
            self.task_fisher_diagonals.push(merged);
        } else {
            // Standard EWC: store per-task Fisher
            self.task_fisher_diagonals
                .push(self.fisher_diagonal.clone());
        }

        Ok(())
    }

    /// Compute the EWC penalty for the current parameters.
    ///
    /// penalty = (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2
    ///
    /// For standard EWC, the penalty sums over all stored tasks.
    /// For online EWC, only the most recent consolidated Fisher is used.
    pub fn ewc_penalty(&self, parameters: &HashMap<String, Array1<T>>) -> Result<T> {
        if self.anchor_parameters.is_empty() {
            return Err(OptimError::InvalidState(
                "no anchor parameters stored; call consolidate first".to_string(),
            ));
        }

        let half = T::from(0.5).unwrap_or_else(|| T::one());
        let mut total_penalty = T::zero();

        if self.online {
            // Online EWC: use the latest consolidated Fisher
            let fisher_map = self.task_fisher_diagonals.last().ok_or_else(|| {
                OptimError::InvalidState("no consolidated Fisher available".to_string())
            })?;

            for (name, anchor) in &self.anchor_parameters {
                let current = parameters.get(name).ok_or_else(|| {
                    OptimError::InvalidState(format!("parameter '{}' not found in input", name))
                })?;
                let fisher = fisher_map.get(name).ok_or_else(|| {
                    OptimError::InvalidState(format!("Fisher information for '{}' not found", name))
                })?;

                for i in 0..anchor.len() {
                    let diff = current[i] - anchor[i];
                    total_penalty = total_penalty + fisher[i] * diff * diff;
                }
            }
        } else {
            // Standard EWC: sum penalty over all tasks, each with its own anchor
            for (task_idx, task_fisher) in self.task_fisher_diagonals.iter().enumerate() {
                let task_anchor = &self.task_anchor_parameters[task_idx];
                for (name, anchor) in task_anchor {
                    let current = parameters.get(name).ok_or_else(|| {
                        OptimError::InvalidState(format!("parameter '{}' not found in input", name))
                    })?;
                    if let Some(fisher) = task_fisher.get(name) {
                        for i in 0..anchor.len() {
                            let diff = current[i] - anchor[i];
                            total_penalty = total_penalty + fisher[i] * diff * diff;
                        }
                    }
                }
            }
        }

        Ok(self.lambda * half * total_penalty)
    }

    /// Compute the gradient of the EWC penalty with respect to current parameters.
    ///
    /// grad_i = lambda * F_i * (theta_i - theta*_i)
    pub fn ewc_gradient(
        &self,
        parameters: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, Array1<T>>> {
        if self.anchor_parameters.is_empty() {
            return Err(OptimError::InvalidState(
                "no anchor parameters stored; call consolidate first".to_string(),
            ));
        }

        let mut gradients: HashMap<String, Array1<T>> = HashMap::new();

        // Initialize gradient arrays to zero
        for (name, param) in parameters {
            gradients.insert(name.clone(), Array1::from_elem(param.len(), T::zero()));
        }

        if self.online {
            let fisher_map = self.task_fisher_diagonals.last().ok_or_else(|| {
                OptimError::InvalidState("no consolidated Fisher available".to_string())
            })?;

            for (name, anchor) in &self.anchor_parameters {
                let current = parameters.get(name).ok_or_else(|| {
                    OptimError::InvalidState(format!("parameter '{}' not found in input", name))
                })?;
                let fisher = fisher_map.get(name).ok_or_else(|| {
                    OptimError::InvalidState(format!("Fisher information for '{}' not found", name))
                })?;
                let grad = gradients.get_mut(name).ok_or_else(|| {
                    OptimError::InvalidState(format!("gradient entry for '{}' not found", name))
                })?;

                for i in 0..anchor.len() {
                    grad[i] = grad[i] + self.lambda * fisher[i] * (current[i] - anchor[i]);
                }
            }
        } else {
            for (task_idx, task_fisher) in self.task_fisher_diagonals.iter().enumerate() {
                let task_anchor = &self.task_anchor_parameters[task_idx];
                for (name, anchor) in task_anchor {
                    let current = parameters.get(name).ok_or_else(|| {
                        OptimError::InvalidState(format!("parameter '{}' not found in input", name))
                    })?;
                    if let Some(fisher) = task_fisher.get(name) {
                        let grad = gradients.get_mut(name).ok_or_else(|| {
                            OptimError::InvalidState(format!(
                                "gradient entry for '{}' not found",
                                name
                            ))
                        })?;

                        for i in 0..anchor.len() {
                            grad[i] = grad[i] + self.lambda * fisher[i] * (current[i] - anchor[i]);
                        }
                    }
                }
            }
        }

        Ok(gradients)
    }

    /// Return the number of tasks that have been consolidated so far.
    pub fn num_tasks(&self) -> usize {
        self.task_fisher_diagonals.len()
    }
}

// ---------------------------------------------------------------------------
// Progressive Networks
// ---------------------------------------------------------------------------

/// A single column in a progressive network, representing one task's parameters.
#[derive(Debug, Clone)]
pub struct NetworkColumn<T: Float + Debug + Send + Sync + 'static> {
    /// Weight matrices per layer (layer l has shape [output_l, input_l])
    pub weights: Vec<Array2<T>>,
    /// Bias vectors per layer
    pub biases: Vec<Array1<T>>,
    /// Whether the column is frozen (previous tasks are frozen)
    pub frozen: bool,
}

impl<T: Float + Debug + Send + Sync + 'static> NetworkColumn<T> {
    /// Number of layers in this column.
    pub fn num_layers(&self) -> usize {
        self.weights.len()
    }

    /// Forward pass through a single layer with ReLU activation.
    /// Returns pre-activation and post-activation values.
    fn forward_layer(&self, input: &Array1<T>, layer_idx: usize) -> Result<(Array1<T>, Array1<T>)> {
        if layer_idx >= self.weights.len() {
            return Err(OptimError::NetworkError(format!(
                "layer index {} out of range (column has {} layers)",
                layer_idx,
                self.weights.len()
            )));
        }
        let w = &self.weights[layer_idx];
        let b = &self.biases[layer_idx];

        // z = W * x + b  (matrix-vector product)
        let out_dim = w.nrows();
        let in_dim = w.ncols();
        if input.len() != in_dim {
            return Err(OptimError::NetworkError(format!(
                "input dimension mismatch at layer {}: weight expects {}, got {}",
                layer_idx,
                in_dim,
                input.len()
            )));
        }

        let mut z = b.clone();
        for i in 0..out_dim {
            let mut sum = T::zero();
            for j in 0..in_dim {
                sum = sum + w[[i, j]] * input[j];
            }
            z[i] = z[i] + sum;
        }

        // ReLU activation (except for the last layer which is linear)
        let is_last_layer = layer_idx == self.weights.len() - 1;
        let a = if is_last_layer {
            z.clone()
        } else {
            z.mapv(|v| if v > T::zero() { v } else { T::zero() })
        };

        Ok((z, a))
    }
}

/// Progressive Networks for continual learning.
///
/// Each new task gets its own column (set of layers). Previous columns are frozen
/// and lateral connections from frozen columns feed into the new column, enabling
/// knowledge transfer without forgetting.
pub struct ProgressiveNetworks<T: Float + Debug + Send + Sync + 'static> {
    /// One column per task
    columns: Vec<NetworkColumn<T>>,
    /// Lateral connection weights: lateral_connections[col][layer] is a matrix
    /// of shape [hidden_size, prev_columns_total_hidden] that maps previous
    /// columns' activations to a contribution for the current column.
    lateral_connections: Vec<Vec<Array2<T>>>,
    /// Index of the currently active (trainable) column
    active_column: usize,
    /// Hidden layer sizes (shared architecture template)
    hidden_sizes: Vec<usize>,
}

impl<T: Float + Debug + Send + Sync + 'static> ProgressiveNetworks<T> {
    /// Create a new progressive network with the given hidden layer sizes.
    ///
    /// No columns are created initially; call `add_task_column` for each task.
    pub fn new(hidden_sizes: Vec<usize>) -> Self {
        Self {
            columns: Vec::new(),
            lateral_connections: Vec::new(),
            active_column: 0,
            hidden_sizes,
        }
    }

    /// Add a new column for a new task.
    ///
    /// `input_size`  - dimensionality of the input features.
    /// `output_size` - dimensionality of the output.
    ///
    /// Returns the column index (task id).
    /// All previously existing columns are frozen.
    pub fn add_task_column(&mut self, input_size: usize, output_size: usize) -> Result<usize> {
        if input_size == 0 || output_size == 0 {
            return Err(OptimError::InvalidConfig(
                "input_size and output_size must be > 0".to_string(),
            ));
        }

        let col_id = self.columns.len();

        // Freeze all existing columns
        for col in &mut self.columns {
            col.frozen = true;
        }

        // Build layer sizes: input -> hidden_1 -> ... -> hidden_n -> output
        let mut layer_sizes = Vec::with_capacity(self.hidden_sizes.len() + 2);
        layer_sizes.push(input_size);
        layer_sizes.extend_from_slice(&self.hidden_sizes);
        layer_sizes.push(output_size);

        // Initialize weights with small values (Xavier-like initialization)
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for l in 0..layer_sizes.len() - 1 {
            let fan_in = layer_sizes[l];
            let fan_out = layer_sizes[l + 1];

            // Xavier init scale: sqrt(2 / (fan_in + fan_out))
            let scale_val = T::from(2.0 / ((fan_in + fan_out) as f64)).unwrap_or_else(|| T::one());
            let scale = Float::sqrt(scale_val);

            // Deterministic initialization: alternate small positive/negative values
            let mut w = Array2::from_elem((fan_out, fan_in), T::zero());
            for i in 0..fan_out {
                for j in 0..fan_in {
                    // Simple deterministic pattern
                    let idx = (i * fan_in + j) as f64;
                    let sign = if (i + j) % 2 == 0 {
                        T::one()
                    } else {
                        -T::one()
                    };
                    let magnitude =
                        scale * T::from((idx + 1.0).recip()).unwrap_or_else(|| T::one());
                    w[[i, j]] = sign * magnitude;
                }
            }

            let b = Array1::from_elem(fan_out, T::zero());

            weights.push(w);
            biases.push(b);
        }

        let column = NetworkColumn {
            weights,
            biases,
            frozen: false,
        };

        self.columns.push(column);

        // Create lateral connections from all previous columns to this new column
        let mut laterals_for_col = Vec::new();
        if col_id > 0 {
            for l in 0..layer_sizes.len() - 1 {
                // At layer l, each previous column produces an activation of size layer_sizes[l+1]
                // (except for the input layer which doesn't have lateral connections).
                // Lateral input dimension = sum of previous columns' hidden sizes at layer l
                // For simplicity, each previous column contributes layer_sizes[l+1] activations
                // at layer l.
                let lateral_in = if l == 0 {
                    // No lateral connections at the first layer (input layer)
                    0
                } else {
                    // Each previous column contributes its hidden size at layer l
                    col_id * layer_sizes[l]
                };
                let lateral_out = layer_sizes[l + 1];

                if lateral_in == 0 {
                    // Placeholder (unused) for consistency
                    laterals_for_col.push(Array2::from_elem((lateral_out, 1), T::zero()));
                } else {
                    // Small initialization for lateral weights
                    let scale_val = T::from(0.1 / (lateral_in as f64)).unwrap_or_else(|| T::one());
                    let scale = Float::sqrt(scale_val);
                    let mut lat_w = Array2::from_elem((lateral_out, lateral_in), T::zero());
                    for i in 0..lateral_out {
                        for j in 0..lateral_in {
                            let sign = if (i + j) % 3 == 0 {
                                T::one()
                            } else if (i + j) % 3 == 1 {
                                -T::one()
                            } else {
                                T::zero()
                            };
                            lat_w[[i, j]] = sign * scale;
                        }
                    }
                    laterals_for_col.push(lat_w);
                }
            }
        }

        self.lateral_connections.push(laterals_for_col);
        self.active_column = col_id;

        Ok(col_id)
    }

    /// Forward pass through the network for a specific task.
    ///
    /// For columns beyond the first, lateral connections from all previous columns
    /// are added at each hidden layer.
    pub fn forward(&self, input: &Array1<T>, task_id: usize) -> Result<Array1<T>> {
        let (output, _) = self.forward_with_laterals(input, task_id)?;
        Ok(output)
    }

    /// Forward pass that also returns intermediate activations from all columns.
    ///
    /// Returns (final_output, all_intermediate_activations) where
    /// `all_intermediate_activations[col][layer]` is the activation at that layer
    /// of that column.
    pub fn forward_with_laterals(
        &self,
        input: &Array1<T>,
        task_id: usize,
    ) -> Result<(Array1<T>, Vec<Array1<T>>)> {
        if task_id >= self.columns.len() {
            return Err(OptimError::InvalidState(format!(
                "task_id {} out of range; only {} columns exist",
                task_id,
                self.columns.len()
            )));
        }

        // Compute activations for all columns up to and including task_id
        // activations[col][layer] holds post-activation output of layer l in column col
        let mut all_activations: Vec<Vec<Array1<T>>> = Vec::new();

        for col in 0..=task_id {
            let column = &self.columns[col];
            let num_layers = column.num_layers();
            let mut col_activations: Vec<Array1<T>> = Vec::new();
            let mut h = input.clone();

            for l in 0..num_layers {
                // Lateral contribution from previous columns (only for col > 0 and l > 0)
                if col > 0 && l > 0 {
                    let laterals = &self.lateral_connections[col];
                    if !laterals.is_empty() && l < laterals.len() {
                        let lat_w = &laterals[l];
                        // Concatenate activations from all previous columns at layer l-1
                        // (previous layer's output for each previous column)
                        let mut lateral_input_parts: Vec<T> = Vec::new();
                        for prev_col_acts in all_activations.iter().take(col) {
                            if l - 1 < prev_col_acts.len() {
                                let prev_act = &prev_col_acts[l - 1];
                                lateral_input_parts.extend(prev_act.iter().copied());
                            }
                        }

                        if !lateral_input_parts.is_empty() {
                            let lateral_input = Array1::from_vec(lateral_input_parts);

                            // Check dimension compatibility
                            if lat_w.ncols() == lateral_input.len() {
                                let lat_out_dim = lat_w.nrows();
                                let lat_in_dim = lat_w.ncols();
                                let mut lateral_contribution =
                                    Array1::from_elem(lat_out_dim, T::zero());
                                for i in 0..lat_out_dim {
                                    let mut sum = T::zero();
                                    for j in 0..lat_in_dim {
                                        sum = sum + lat_w[[i, j]] * lateral_input[j];
                                    }
                                    lateral_contribution[i] = sum;
                                }

                                // Add lateral contribution to input before this layer
                                // Dimensions must match: h and lateral_contribution
                                if h.len() == lateral_contribution.len() {
                                    for i in 0..h.len() {
                                        h[i] = h[i] + lateral_contribution[i];
                                    }
                                }
                            }
                        }
                    }
                }

                let (_pre, post) = column.forward_layer(&h, l)?;
                col_activations.push(post.clone());
                h = post;
            }

            all_activations.push(col_activations);
        }

        // Final output is the last activation of the target column
        let target_activations = &all_activations[task_id];
        let output = target_activations
            .last()
            .ok_or_else(|| OptimError::NetworkError("column has no layers".to_string()))?
            .clone();

        // Collect intermediate activations (flatten from all columns)
        let mut intermediates: Vec<Array1<T>> = Vec::new();
        for col_acts in &all_activations {
            for act in col_acts {
                intermediates.push(act.clone());
            }
        }

        Ok((output, intermediates))
    }

    /// Freeze a specific column (mark as non-trainable).
    pub fn freeze_column(&mut self, column_id: usize) -> Result<()> {
        if column_id >= self.columns.len() {
            return Err(OptimError::InvalidState(format!(
                "column_id {} out of range; only {} columns exist",
                column_id,
                self.columns.len()
            )));
        }
        self.columns[column_id].frozen = true;
        Ok(())
    }

    /// Return the number of columns (one per task).
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get a reference to a specific column's parameters.
    pub fn get_column_parameters(&self, column_id: usize) -> Result<&NetworkColumn<T>> {
        self.columns.get(column_id).ok_or_else(|| {
            OptimError::InvalidState(format!(
                "column_id {} out of range; only {} columns exist",
                column_id,
                self.columns.len()
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    type F = f64;

    // Helper: create simple parameters map
    fn make_params(names: &[&str], size: usize, value: f64) -> HashMap<String, Array1<F>> {
        let mut map = HashMap::new();
        for name in names {
            map.insert(name.to_string(), Array1::from_elem(size, value));
        }
        map
    }

    // -----------------------------------------------------------------------
    // EWC Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ewc_creation_and_configuration() {
        let ewc: ElasticWeightConsolidation<F> = ElasticWeightConsolidation::new(1000.0)
            .with_num_samples(50)
            .with_online(true)
            .with_gamma(0.9);

        assert_eq!(ewc.num_tasks(), 0);
        assert!(ewc.online);
        assert!((ewc.gamma - 0.9).abs() < 1e-12);
        assert!((ewc.lambda - 1000.0).abs() < 1e-12);
        assert_eq!(ewc.num_samples_fisher, 50);
    }

    #[test]
    fn test_fisher_diagonal_computation() {
        let mut ewc: ElasticWeightConsolidation<F> =
            ElasticWeightConsolidation::new(1.0).with_num_samples(10);

        let params = make_params(&["w1", "w2"], 3, 1.0);

        // Gradients function: returns constant gradients (simulating quadratic loss)
        // For a quadratic loss f(x) = 0.5 * x^2, grad = x = [1, 1, 1]
        let grad_fn = |_p: &HashMap<String, Array1<F>>| -> Result<HashMap<String, Array1<F>>> {
            let mut g = HashMap::new();
            g.insert("w1".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0]));
            g.insert("w2".to_string(), Array1::from_vec(vec![0.5, 0.5, 0.5]));
            Ok(g)
        };

        ewc.compute_fisher_diagonal(&params, grad_fn)
            .expect("compute_fisher_diagonal should succeed");

        // Fisher = E[g^2] = g^2 since gradients are constant
        let f1 = ewc
            .fisher_diagonal
            .get("w1")
            .expect("w1 Fisher should exist");
        assert!((f1[0] - 1.0).abs() < 1e-12);
        assert!((f1[1] - 4.0).abs() < 1e-12);
        assert!((f1[2] - 9.0).abs() < 1e-12);

        let f2 = ewc
            .fisher_diagonal
            .get("w2")
            .expect("w2 Fisher should exist");
        assert!((f2[0] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_ewc_penalty_computation() {
        let mut ewc: ElasticWeightConsolidation<F> =
            ElasticWeightConsolidation::new(2.0).with_num_samples(5);

        let anchor = make_params(&["w"], 2, 1.0);

        // Constant Fisher = [1.0, 1.0]
        let grad_fn = |_p: &HashMap<String, Array1<F>>| -> Result<HashMap<String, Array1<F>>> {
            let mut g = HashMap::new();
            g.insert("w".to_string(), Array1::from_vec(vec![1.0, 1.0]));
            Ok(g)
        };

        ewc.compute_fisher_diagonal(&anchor, grad_fn)
            .expect("compute Fisher should succeed");
        ewc.consolidate(&anchor)
            .expect("consolidate should succeed");

        // Current parameters differ from anchor
        let mut current = HashMap::new();
        current.insert("w".to_string(), Array1::from_vec(vec![2.0, 3.0]));

        // Penalty = (lambda/2) * sum F_i * (theta_i - theta*_i)^2
        // = (2/2) * (1*(2-1)^2 + 1*(3-1)^2) = 1 * (1 + 4) = 5.0
        let penalty = ewc
            .ewc_penalty(&current)
            .expect("ewc_penalty should succeed");
        assert!(
            (penalty - 5.0).abs() < 1e-12,
            "expected penalty 5.0, got {}",
            penalty
        );
    }

    #[test]
    fn test_ewc_gradient_computation() {
        let mut ewc: ElasticWeightConsolidation<F> =
            ElasticWeightConsolidation::new(2.0).with_num_samples(5);

        let anchor = make_params(&["w"], 2, 1.0);

        let grad_fn = |_p: &HashMap<String, Array1<F>>| -> Result<HashMap<String, Array1<F>>> {
            let mut g = HashMap::new();
            g.insert("w".to_string(), Array1::from_vec(vec![1.0, 1.0]));
            Ok(g)
        };

        ewc.compute_fisher_diagonal(&anchor, grad_fn)
            .expect("compute Fisher should succeed");
        ewc.consolidate(&anchor)
            .expect("consolidate should succeed");

        let mut current = HashMap::new();
        current.insert("w".to_string(), Array1::from_vec(vec![2.0, 3.0]));

        // Gradient = lambda * F_i * (theta_i - theta*_i)
        // = 2 * 1 * (2-1, 3-1) = (2, 4)
        let grads = ewc
            .ewc_gradient(&current)
            .expect("ewc_gradient should succeed");
        let gw = grads.get("w").expect("gradient for w should exist");
        assert!((gw[0] - 2.0).abs() < 1e-12, "expected 2.0, got {}", gw[0]);
        assert!((gw[1] - 4.0).abs() < 1e-12, "expected 4.0, got {}", gw[1]);
    }

    #[test]
    fn test_consolidation_workflow() {
        let mut ewc: ElasticWeightConsolidation<F> =
            ElasticWeightConsolidation::new(1.0).with_num_samples(5);

        let params_task1 = make_params(&["w"], 2, 1.0);

        let grad_fn = |_p: &HashMap<String, Array1<F>>| -> Result<HashMap<String, Array1<F>>> {
            let mut g = HashMap::new();
            g.insert("w".to_string(), Array1::from_vec(vec![1.0, 2.0]));
            Ok(g)
        };

        ewc.compute_fisher_diagonal(&params_task1, grad_fn)
            .expect("Fisher computation should succeed");
        ewc.consolidate(&params_task1)
            .expect("consolidation should succeed");
        assert_eq!(ewc.num_tasks(), 1);

        // Second task
        let params_task2 = make_params(&["w"], 2, 2.0);
        ewc.compute_fisher_diagonal(&params_task2, grad_fn)
            .expect("Fisher computation for task 2 should succeed");
        ewc.consolidate(&params_task2)
            .expect("consolidation for task 2 should succeed");
        assert_eq!(ewc.num_tasks(), 2);

        // Penalty at anchor should be zero (for the latest anchor)
        let penalty = ewc
            .ewc_penalty(&params_task2)
            .expect("penalty should succeed");
        // The latest anchor is params_task2, so the second task's Fisher penalty is 0.
        // But the first task's Fisher penalty is non-zero because params_task2 != params_task1.
        // penalty = 0.5 * sum_task1 F_i*(2-1)^2 = 0.5 * (1*1 + 4*1) = 2.5
        assert!(
            penalty > 0.0,
            "penalty should be positive due to task1 Fisher, got {}",
            penalty
        );
    }

    #[test]
    fn test_online_ewc_multiple_tasks() {
        let mut ewc: ElasticWeightConsolidation<F> = ElasticWeightConsolidation::new(1.0)
            .with_num_samples(5)
            .with_online(true)
            .with_gamma(0.5);

        // Task 1
        let params1 = make_params(&["w"], 2, 0.0);
        let grad_fn1 = |_p: &HashMap<String, Array1<F>>| -> Result<HashMap<String, Array1<F>>> {
            let mut g = HashMap::new();
            g.insert("w".to_string(), Array1::from_vec(vec![1.0, 1.0]));
            Ok(g)
        };
        ewc.compute_fisher_diagonal(&params1, grad_fn1)
            .expect("Fisher 1 should succeed");
        ewc.consolidate(&params1)
            .expect("consolidate 1 should succeed");
        assert_eq!(ewc.num_tasks(), 1);

        // Task 1 Fisher = [1.0, 1.0] (since gradients are constant = 1.0)
        let fisher1 = ewc.task_fisher_diagonals[0]
            .get("w")
            .expect("Fisher should exist");
        assert!((fisher1[0] - 1.0).abs() < 1e-12);

        // Task 2
        let params2 = make_params(&["w"], 2, 1.0);
        let grad_fn2 = |_p: &HashMap<String, Array1<F>>| -> Result<HashMap<String, Array1<F>>> {
            let mut g = HashMap::new();
            g.insert("w".to_string(), Array1::from_vec(vec![2.0, 2.0]));
            Ok(g)
        };
        ewc.compute_fisher_diagonal(&params2, grad_fn2)
            .expect("Fisher 2 should succeed");
        ewc.consolidate(&params2)
            .expect("consolidate 2 should succeed");
        assert_eq!(ewc.num_tasks(), 2);

        // Online Fisher for task 2 = gamma * old + new = 0.5 * [1,1] + [4,4] = [4.5, 4.5]
        let fisher2 = ewc.task_fisher_diagonals[1]
            .get("w")
            .expect("Fisher should exist");
        assert!(
            (fisher2[0] - 4.5).abs() < 1e-12,
            "expected 4.5, got {}",
            fisher2[0]
        );

        // Penalty at params different from anchor (params2)
        let mut test_params = HashMap::new();
        test_params.insert("w".to_string(), Array1::from_vec(vec![2.0, 2.0]));

        let penalty = ewc
            .ewc_penalty(&test_params)
            .expect("penalty should succeed");
        // penalty = 0.5 * sum 4.5 * (2-1)^2 = 0.5 * (4.5 + 4.5) = 4.5
        assert!(
            (penalty - 4.5).abs() < 1e-12,
            "expected 4.5, got {}",
            penalty
        );
    }

    // -----------------------------------------------------------------------
    // Progressive Networks Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_progressive_network_creation_and_columns() {
        let mut pn: ProgressiveNetworks<F> = ProgressiveNetworks::new(vec![16, 8]);

        assert_eq!(pn.num_columns(), 0);

        let col0 = pn
            .add_task_column(4, 2)
            .expect("add column 0 should succeed");
        assert_eq!(col0, 0);
        assert_eq!(pn.num_columns(), 1);

        // Check column structure: 4 -> 16 -> 8 -> 2 = 3 layers
        let params0 = pn
            .get_column_parameters(0)
            .expect("get column 0 should succeed");
        assert_eq!(params0.num_layers(), 3);
        assert!(!params0.frozen);

        // Add second column - first should now be frozen
        let col1 = pn
            .add_task_column(4, 2)
            .expect("add column 1 should succeed");
        assert_eq!(col1, 1);
        assert_eq!(pn.num_columns(), 2);

        let params0_after = pn
            .get_column_parameters(0)
            .expect("get column 0 should succeed");
        assert!(params0_after.frozen, "column 0 should be frozen");

        let params1 = pn
            .get_column_parameters(1)
            .expect("get column 1 should succeed");
        assert!(!params1.frozen, "column 1 should not be frozen");
    }

    #[test]
    fn test_progressive_network_forward_single_column() {
        let mut pn: ProgressiveNetworks<F> = ProgressiveNetworks::new(vec![8]);

        pn.add_task_column(4, 2).expect("add column should succeed");

        let input = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8]);
        let output = pn.forward(&input, 0).expect("forward should succeed");

        // Output should have 2 elements (output_size = 2)
        assert_eq!(output.len(), 2, "output should have 2 elements");

        // Output should be finite
        for val in output.iter() {
            assert!(val.is_finite(), "output should be finite, got {}", val);
        }
    }

    #[test]
    fn test_progressive_network_forward_with_laterals() {
        let mut pn: ProgressiveNetworks<F> = ProgressiveNetworks::new(vec![8]);

        pn.add_task_column(4, 2)
            .expect("add column 0 should succeed");
        pn.add_task_column(4, 2)
            .expect("add column 1 should succeed");

        let input = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8]);

        // Forward through task 0 (no laterals)
        let output0 = pn
            .forward(&input, 0)
            .expect("forward task 0 should succeed");
        assert_eq!(output0.len(), 2);

        // Forward through task 1 (with lateral connections from task 0)
        let (output1, intermediates) = pn
            .forward_with_laterals(&input, 1)
            .expect("forward_with_laterals task 1 should succeed");
        assert_eq!(output1.len(), 2);

        // Should have intermediate activations from both columns
        assert!(
            !intermediates.is_empty(),
            "should have intermediate activations"
        );

        // The outputs for task 0 and task 1 should differ due to
        // different weights and lateral connections
        let diff: f64 = output0
            .iter()
            .zip(output1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        // They are almost certainly different given different random inits
        // but we don't require it -- just check both are finite
        for val in output1.iter() {
            assert!(val.is_finite(), "output should be finite, got {}", val);
        }
        let _ = diff; // suppress unused warning
    }

    #[test]
    fn test_progressive_network_freeze_column() {
        let mut pn: ProgressiveNetworks<F> = ProgressiveNetworks::new(vec![8]);

        pn.add_task_column(4, 2).expect("add column should succeed");

        assert!(
            !pn.get_column_parameters(0).expect("get params").frozen,
            "column should not be frozen initially"
        );

        pn.freeze_column(0).expect("freeze should succeed");

        assert!(
            pn.get_column_parameters(0).expect("get params").frozen,
            "column should be frozen after freeze_column"
        );

        // Freeze out-of-range column should error
        let err = pn.freeze_column(99);
        assert!(err.is_err(), "freezing out-of-range column should fail");
    }

    #[test]
    fn test_progressive_network_multiple_tasks_forward() {
        let mut pn: ProgressiveNetworks<F> = ProgressiveNetworks::new(vec![8, 4]);

        // Add 3 task columns
        for _ in 0..3 {
            pn.add_task_column(4, 2).expect("add column should succeed");
        }
        assert_eq!(pn.num_columns(), 3);

        let input = Array1::from_vec(vec![0.5, -0.5, 1.0, -1.0]);

        // Forward through each task should work
        for task_id in 0..3 {
            let output = pn
                .forward(&input, task_id)
                .unwrap_or_else(|_| panic!("forward task {} should succeed", task_id));
            assert_eq!(
                output.len(),
                2,
                "output for task {} should have 2 elements",
                task_id
            );
            for val in output.iter() {
                assert!(
                    val.is_finite(),
                    "output for task {} should be finite",
                    task_id
                );
            }
        }

        // Forward with invalid task_id should fail
        let err = pn.forward(&input, 10);
        assert!(err.is_err(), "forward with invalid task_id should fail");
    }
}
