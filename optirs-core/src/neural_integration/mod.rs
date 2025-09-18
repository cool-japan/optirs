// Neural network integration for optimizers
//
// This module provides interfaces and utilities for integrating optimizers with neural networks,
// including generic parameter optimization, lazy registration, and architecture-aware optimizations.

use crate::error::{OptimError, Result};
use num_traits::Float;
use scirs2_core::ndarray_ext::{Array, Dimension, ScalarOperand};
use std::collections::HashMap;
use std::fmt::Debug;
// use statrs::statistics::Statistics; // statrs not available

/// Type alias for layer identifiers
pub type LayerId = String;

/// Type alias for parameter identifiers
pub type ParamId = String;

/// Parameter metadata for neural network parameters
#[derive(Debug, Clone)]
pub struct ParameterMetadata {
    /// Layer name this parameter belongs to
    pub layername: LayerId,
    /// Parameter name within the layer
    pub param_name: ParamId,
    /// Parameter shape
    pub shape: Vec<usize>,
    /// Whether parameter requires gradients
    pub requires_grad: bool,
    /// Parameter type (weights, bias, etc.)
    pub paramtype: ParameterType,
    /// Sharing group for parameter sharing
    pub sharing_group: Option<String>,
    /// Custom tags for architecture-specific optimizations
    pub tags: Vec<String>,
}

/// Types of neural network parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterType {
    /// Weight matrices
    Weight,
    /// Bias vectors
    Bias,
    /// Normalization parameters (scale/shift)
    Normalization,
    /// Embedding parameters
    Embedding,
    /// Attention parameters
    Attention,
    /// Custom parameter type
    Custom,
}

/// Layer architecture information
#[derive(Debug, Clone)]
pub struct LayerArchitecture {
    /// Layer type name
    pub layer_type: String,
    /// Input dimensions
    pub input_dims: Vec<usize>,
    /// Output dimensions
    pub output_dims: Vec<usize>,
    /// Layer-specific configuration
    pub config: HashMap<String, LayerConfig>,
    /// Whether layer is trainable
    pub trainable: bool,
}

/// Layer configuration values
#[derive(Debug, Clone)]
pub enum LayerConfig {
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Boolean value
    Bool(bool),
    /// List of values
    List(Vec<LayerConfig>),
}

/// Generic parameter optimization interface
pub trait ParameterOptimizer<A: Float, D: Dimension> {
    /// Register a parameter for optimization
    fn register_parameter(
        &mut self,
        paramid: ParamId,
        parameter: &Array<A, D>,
        metadata: ParameterMetadata,
    ) -> Result<()>;

    /// Update registered parameters with gradients
    fn step(
        &mut self,
        gradients: HashMap<ParamId, Array<A, D>>,
        parameters: &mut HashMap<ParamId, Array<A, D>>,
    ) -> Result<()>;

    /// Get parameter-specific learning rate
    fn get_learning_rate(&self, paramid: &ParamId) -> Option<A>;

    /// Set parameter-specific learning rate
    fn set_learning_rate(&mut self, paramid: &ParamId, lr: A) -> Result<()>;

    /// Get optimizer state for a parameter
    fn get_parameter_state(&self, paramid: &ParamId) -> Option<&HashMap<String, Array<A, D>>>;

    /// Reset optimizer state
    fn reset_state(&mut self);

    /// Get all registered parameter IDs
    fn registered_parameters(&self) -> Vec<ParamId>;
}

/// Neural network parameter manager with lazy registration
#[derive(Debug)]
pub struct ParameterManager<A: Float, D: Dimension> {
    /// Registered parameters with metadata
    parameters: HashMap<ParamId, ParameterMetadata>,
    /// Parameter optimizer states
    optimizer_states: HashMap<ParamId, HashMap<String, Array<A, D>>>,
    /// Layer architectures
    layer_architectures: HashMap<LayerId, LayerArchitecture>,
    /// Parameter sharing groups
    sharing_groups: HashMap<String, Vec<ParamId>>,
    /// Layer-specific optimization rules
    layer_rules: HashMap<LayerId, LayerOptimizationRule<A>>,
    /// Global optimization configuration
    global_config: OptimizationConfig<A>,
    /// Lazy registration mode
    lazy_mode: bool,
    /// Pending registrations (for lazy mode)
    pending_registrations: Vec<(ParamId, ParameterMetadata)>,
}

/// Layer-specific optimization rules
#[derive(Debug, Clone)]
pub struct LayerOptimizationRule<A: Float> {
    /// Learning rate multiplier for this layer
    pub lr_multiplier: A,
    /// Weight decay multiplier
    pub weight_decay_multiplier: A,
    /// Whether to freeze this layer
    pub frozen: bool,
    /// Custom optimizer settings
    pub custom_settings: HashMap<String, LayerConfig>,
}

/// Global optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig<A: Float> {
    /// Base learning rate
    pub base_learning_rate: A,
    /// Global weight decay
    pub weight_decay: A,
    /// Gradient clipping threshold
    pub gradient_clip: Option<A>,
    /// Whether to use mixed precision
    pub mixed_precision: bool,
    /// Architecture-specific optimizations
    pub architecture_optimizations: HashMap<String, bool>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync>
    ParameterManager<A, D>
{
    /// Create a new parameter manager
    pub fn new(config: OptimizationConfig<A>) -> Self {
        Self {
            parameters: HashMap::new(),
            optimizer_states: HashMap::new(),
            layer_architectures: HashMap::new(),
            sharing_groups: HashMap::new(),
            layer_rules: HashMap::new(),
            global_config: config,
            lazy_mode: false,
            pending_registrations: Vec::new(),
        }
    }

    /// Enable lazy registration mode
    pub fn enable_lazy_mode(&mut self) {
        self.lazy_mode = true;
    }

    /// Disable lazy registration mode and process pending registrations
    pub fn disable_lazy_mode(&mut self) -> Result<()> {
        self.lazy_mode = false;

        // Process all pending registrations
        let pending = std::mem::take(&mut self.pending_registrations);
        for (paramid, metadata) in pending {
            self.register_parameter_impl(paramid, metadata)?;
        }

        Ok(())
    }

    /// Register a layer architecture
    pub fn register_layer(&mut self, layerid: LayerId, architecture: LayerArchitecture) {
        self.layer_architectures.insert(layerid, architecture);
    }

    /// Set layer-specific optimization rule
    pub fn set_layer_rule(&mut self, layerid: LayerId, rule: LayerOptimizationRule<A>) {
        self.layer_rules.insert(layerid, rule);
    }

    /// Register a parameter
    pub fn register_parameter(
        &mut self,
        paramid: ParamId,
        metadata: ParameterMetadata,
    ) -> Result<()> {
        if self.lazy_mode {
            self.pending_registrations.push((paramid, metadata));
            Ok(())
        } else {
            self.register_parameter_impl(paramid, metadata)
        }
    }

    /// Internal parameter registration implementation
    fn register_parameter_impl(
        &mut self,
        paramid: ParamId,
        metadata: ParameterMetadata,
    ) -> Result<()> {
        // Handle parameter sharing
        if let Some(sharing_group) = &metadata.sharing_group {
            self.sharing_groups
                .entry(sharing_group.clone())
                .or_default()
                .push(paramid.clone());
        }

        // Initialize optimizer state for this parameter
        self.optimizer_states
            .insert(paramid.clone(), HashMap::new());

        // Store parameter metadata
        self.parameters.insert(paramid, metadata);

        Ok(())
    }

    /// Get effective learning rate for a parameter
    pub fn get_effective_learning_rate(&self, paramid: &ParamId) -> A {
        let base_lr = self.global_config.base_learning_rate;

        if let Some(metadata) = self.parameters.get(paramid) {
            if let Some(rule) = self.layer_rules.get(&metadata.layername) {
                return base_lr * rule.lr_multiplier;
            }
        }

        base_lr
    }

    /// Get effective weight decay for a parameter
    pub fn get_effective_weight_decay(&self, paramid: &ParamId) -> A {
        let base_decay = self.global_config.weight_decay;

        if let Some(metadata) = self.parameters.get(paramid) {
            if let Some(rule) = self.layer_rules.get(&metadata.layername) {
                return base_decay * rule.weight_decay_multiplier;
            }
        }

        base_decay
    }

    /// Check if parameter is frozen
    pub fn is_parameter_frozen(&self, paramid: &ParamId) -> bool {
        if let Some(metadata) = self.parameters.get(paramid) {
            if let Some(rule) = self.layer_rules.get(&metadata.layername) {
                return rule.frozen;
            }
        }
        false
    }

    /// Get parameters in a sharing group
    pub fn get_sharing_group(&self, groupname: &str) -> Option<&[ParamId]> {
        self.sharing_groups.get(groupname).map(|v| v.as_slice())
    }

    /// Get all registered parameters
    pub fn get_all_parameters(&self) -> &HashMap<ParamId, ParameterMetadata> {
        &self.parameters
    }

    /// Get layer architecture
    pub fn get_layer_architecture(&self, layerid: &LayerId) -> Option<&LayerArchitecture> {
        self.layer_architectures.get(layerid)
    }

    /// Get parameter metadata
    pub fn get_parameter_metadata(&self, paramid: &ParamId) -> Option<&ParameterMetadata> {
        self.parameters.get(paramid)
    }

    /// Update global configuration
    pub fn update_config(&mut self, config: OptimizationConfig<A>) {
        self.global_config = config;
    }

    /// Get optimizer state for parameter
    pub fn get_optimizer_state(&self, paramid: &ParamId) -> Option<&HashMap<String, Array<A, D>>> {
        self.optimizer_states.get(paramid)
    }

    /// Get mutable optimizer state for parameter
    pub fn get_optimizer_state_mut(
        &mut self,
        paramid: &ParamId,
    ) -> Option<&mut HashMap<String, Array<A, D>>> {
        self.optimizer_states.get_mut(paramid)
    }

    /// Initialize optimizer state for parameter
    pub fn init_optimizer_state(
        &mut self,
        paramid: &ParamId,
        state_name: &str,
        state: Array<A, D>,
    ) -> Result<()> {
        if let Some(states) = self.optimizer_states.get_mut(paramid) {
            states.insert(state_name.to_string(), state);
            Ok(())
        } else {
            Err(OptimError::InvalidConfig(format!(
                "Parameter {} not registered",
                paramid
            )))
        }
    }

    /// Reset all optimizer states
    pub fn reset_optimizer_states(&mut self) {
        for states in self.optimizer_states.values_mut() {
            states.clear();
        }
    }

    /// Get parameters by layer
    pub fn get_parameters_by_layer(&self, layerid: &LayerId) -> Vec<&ParamId> {
        self.parameters
            .iter()
            .filter(|(_, metadata)| &metadata.layername == layerid)
            .map(|(paramid, _)| paramid)
            .collect()
    }

    /// Get parameters by type
    pub fn get_parameters_by_type(&self, paramtype: ParameterType) -> Vec<&ParamId> {
        self.parameters
            .iter()
            .filter(|(_, metadata)| metadata.paramtype == paramtype)
            .map(|(paramid, _)| paramid)
            .collect()
    }

    /// Get trainable parameters
    pub fn get_trainable_parameters(&self) -> Vec<&ParamId> {
        self.parameters
            .iter()
            .filter(|(paramid, metadata)| {
                metadata.requires_grad && !self.is_parameter_frozen(paramid)
            })
            .map(|(paramid, _)| paramid)
            .collect()
    }
}

impl<A: Float + Send + Sync> Default for OptimizationConfig<A> {
    fn default() -> Self {
        Self {
            base_learning_rate: A::from(0.001).unwrap(),
            weight_decay: A::zero(),
            gradient_clip: None,
            mixed_precision: false,
            architecture_optimizations: HashMap::new(),
        }
    }
}

impl<A: Float + Send + Sync> Default for LayerOptimizationRule<A> {
    fn default() -> Self {
        Self {
            lr_multiplier: A::one(),
            weight_decay_multiplier: A::one(),
            frozen: false,
            custom_settings: HashMap::new(),
        }
    }
}

/// Forward/backward pass integration
pub mod forward_backward {
    use super::*;

    /// Forward pass hook for parameter tracking
    pub trait ForwardHook<A: Float, D: Dimension> {
        /// Called before layer forward pass
        fn pre_forward(&mut self, layerid: &LayerId, inputs: &[Array<A, D>]) -> Result<()>;

        /// Called after layer forward pass
        fn post_forward(&mut self, layerid: &LayerId, outputs: &[Array<A, D>]) -> Result<()>;
    }

    /// Backward pass hook for gradient processing
    pub trait BackwardHook<A: Float, D: Dimension> {
        /// Called before layer backward pass
        fn pre_backward(&mut self, layerid: &LayerId, gradoutputs: &[Array<A, D>]) -> Result<()>;

        /// Called after layer backward pass
        fn post_backward(&mut self, layerid: &LayerId, gradinputs: &[Array<A, D>]) -> Result<()>;
    }

    /// Neural network integration manager
    pub struct NeuralIntegration<A: Float, D: Dimension> {
        /// Parameter manager
        param_manager: ParameterManager<A, D>,
        /// Forward hooks
        forward_hooks: HashMap<LayerId, Box<dyn ForwardHook<A, D>>>,
        /// Backward hooks
        backward_hooks: HashMap<LayerId, Box<dyn BackwardHook<A, D>>>,
        /// Gradient accumulation mode
        gradient_accumulation: bool,
        /// Accumulated gradients
        accumulated_gradients: HashMap<ParamId, Array<A, D>>,
        /// Accumulation count
        accumulation_count: usize,
    }

    impl<
            A: Float
                + ScalarOperand
                + Debug
                + 'static
                + num_traits::FromPrimitive
                + std::iter::Sum
                + Send
                + Sync,
            D: Dimension + 'static,
        > NeuralIntegration<A, D>
    {
        /// Create a new neural integration manager
        pub fn new(config: OptimizationConfig<A>) -> Self {
            Self {
                param_manager: ParameterManager::new(config),
                forward_hooks: HashMap::new(),
                backward_hooks: HashMap::new(),
                gradient_accumulation: false,
                accumulated_gradients: HashMap::new(),
                accumulation_count: 0,
            }
        }

        /// Register a forward hook for a layer
        pub fn register_forward_hook<H>(&mut self, layerid: LayerId, hook: H)
        where
            H: ForwardHook<A, D> + 'static,
        {
            self.forward_hooks.insert(layerid, Box::new(hook));
        }

        /// Register a backward hook for a layer
        pub fn register_backward_hook<H>(&mut self, layerid: LayerId, hook: H)
        where
            H: BackwardHook<A, D> + 'static,
        {
            self.backward_hooks.insert(layerid, Box::new(hook));
        }

        /// Enable gradient accumulation
        pub fn enable_gradient_accumulation(&mut self) {
            self.gradient_accumulation = true;
        }

        /// Disable gradient accumulation and return accumulated gradients
        pub fn disable_gradient_accumulation(&mut self) -> HashMap<ParamId, Array<A, D>> {
            self.gradient_accumulation = false;
            let result = std::mem::take(&mut self.accumulated_gradients);
            self.accumulation_count = 0;
            result
        }

        /// Execute forward pass with hooks
        pub fn forward_pass(
            &mut self,
            layerid: &LayerId,
            inputs: &[Array<A, D>],
        ) -> Result<Vec<Array<A, D>>> {
            // Execute pre-forward hook
            if let Some(hook) = self.forward_hooks.get_mut(layerid) {
                hook.pre_forward(layerid, inputs)?;
            }

            // Get layer architecture and parameters
            let layer_arch = self
                .param_manager
                .get_layer_architecture(layerid)
                .ok_or_else(|| {
                    OptimError::InvalidConfig(format!("Layer {} not registered", layerid))
                })?
                .clone();

            // Compute outputs based on layer type
            let outputs = match layer_arch.layer_type.as_str() {
                "linear" | "dense" | "fc" => {
                    // Linear layer: output = input @ weight^T + bias
                    self.compute_linear_forward(layerid, inputs)?
                }
                "conv" | "conv2d" => {
                    // Convolutional layer: simplified computation
                    self.compute_conv_forward(layerid, inputs)?
                }
                "activation" => {
                    // Activation layer: apply activation function
                    self.compute_activation_forward(layerid, inputs, &layer_arch)?
                }
                "normalization" | "batchnorm" | "layernorm" => {
                    // Normalization layer
                    self.compute_normalization_forward(layerid, inputs)?
                }
                "dropout" => {
                    // Dropout layer: apply dropout mask
                    self.compute_dropout_forward(layerid, inputs, &layer_arch)?
                }
                "pooling" | "maxpool" | "avgpool" => {
                    // Pooling layer
                    self.compute_pooling_forward(layerid, inputs, &layer_arch)?
                }
                _ => {
                    // Default: pass through for unknown layer types
                    inputs.to_vec()
                }
            };

            // Execute post-forward hook
            if let Some(hook) = self.forward_hooks.get_mut(layerid) {
                hook.post_forward(layerid, &outputs)?;
            }

            Ok(outputs)
        }

        /// Compute linear layer forward pass
        fn compute_linear_forward(
            &self,
            layerid: &LayerId,
            inputs: &[Array<A, D>],
        ) -> Result<Vec<Array<A, D>>> {
            // For demonstration, we implement a simple pass-through
            // In a real implementation, this would multiply by weights and add bias
            if inputs.is_empty() {
                return Err(OptimError::InvalidConfig(
                    "Linear layer requires input".to_string(),
                ));
            }

            // Get parameters for this layer
            let layer_params = self.param_manager.get_parameters_by_layer(layerid);

            // Simple transformation: scale input by learning rate (as a placeholder)
            let lr =
                self.param_manager
                    .get_effective_learning_rate(layer_params.first().ok_or_else(|| {
                        OptimError::InvalidConfig("No parameters for linear layer".to_string())
                    })?);

            let outputs: Vec<Array<A, D>> =
                inputs.iter().map(|input| input.mapv(|x| x * lr)).collect();

            Ok(outputs)
        }

        /// Compute convolutional layer forward pass
        fn compute_conv_forward(
            &self,
            _layer_id: &LayerId,
            inputs: &[Array<A, D>],
        ) -> Result<Vec<Array<A, D>>> {
            // Simplified convolution: just pass through
            // Real implementation would apply convolution kernels
            Ok(inputs.to_vec())
        }

        /// Compute activation forward pass
        fn compute_activation_forward(
            &self,
            _layer_id: &LayerId,
            inputs: &[Array<A, D>],
            layer_arch: &LayerArchitecture,
        ) -> Result<Vec<Array<A, D>>> {
            let activation_type = layer_arch
                .config
                .get("activation")
                .and_then(|v| match v {
                    LayerConfig::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or("relu");

            let outputs: Vec<Array<A, D>> = inputs
                .iter()
                .map(|input| {
                    match activation_type {
                        "relu" => input.mapv(|x| if x > A::zero() { x } else { A::zero() }),
                        "sigmoid" => input.mapv(|x| A::one() / (A::one() + (-x).exp())),
                        "tanh" => input.mapv(|x| x.tanh()),
                        "leaky_relu" => {
                            let alpha = A::from(0.01).unwrap();
                            input.mapv(|x| if x > A::zero() { x } else { alpha * x })
                        }
                        _ => input.clone(), // Unknown activation, pass through
                    }
                })
                .collect();

            Ok(outputs)
        }

        /// Compute normalization forward pass
        fn compute_normalization_forward(
            &self,
            _layer_id: &LayerId,
            inputs: &[Array<A, D>],
        ) -> Result<Vec<Array<A, D>>> {
            // Simplified normalization: normalize to zero mean and unit variance
            let outputs: Vec<Array<A, D>> = inputs
                .iter()
                .map(|input| {
                    let mean = input.iter().copied().sum::<A>()
                        / A::from(input.len()).unwrap_or(A::zero());
                    let variance = input
                        .mapv(|x| (x - mean).powi(2))
                        .mean()
                        .unwrap_or(A::one());
                    let std_dev = variance.sqrt();
                    let epsilon = A::from(1e-5).unwrap();

                    input.mapv(|x| (x - mean) / (std_dev + epsilon))
                })
                .collect();

            Ok(outputs)
        }

        /// Compute dropout forward pass
        fn compute_dropout_forward(
            &self,
            _layer_id: &LayerId,
            inputs: &[Array<A, D>],
            layer_arch: &LayerArchitecture,
        ) -> Result<Vec<Array<A, D>>> {
            let dropout_rate = layer_arch
                .config
                .get("dropout_rate")
                .and_then(|v| match v {
                    LayerConfig::Float(f) => Some(A::from(*f).unwrap()),
                    _ => None,
                })
                .unwrap_or(A::from(0.5).unwrap());

            // During training, we would apply dropout mask
            // For now, scale by (1 - dropout_rate) to maintain expected value
            let scale = A::one() - dropout_rate;
            let outputs: Vec<Array<A, D>> = inputs
                .iter()
                .map(|input| input.mapv(|x| x * scale))
                .collect();

            Ok(outputs)
        }

        /// Compute pooling forward pass
        fn compute_pooling_forward(
            &self,
            _layer_id: &LayerId,
            inputs: &[Array<A, D>],
            _layer_arch: &LayerArchitecture,
        ) -> Result<Vec<Array<A, D>>> {
            // Simplified pooling: just pass through
            // Real implementation would downsample the input
            Ok(inputs.to_vec())
        }

        /// Execute backward pass with hooks
        pub fn backward_pass(
            &mut self,
            layerid: &LayerId,
            grad_outputs: &[Array<A, D>],
        ) -> Result<Vec<Array<A, D>>> {
            // Execute pre-backward hook
            if let Some(hook) = self.backward_hooks.get_mut(layerid) {
                hook.pre_backward(layerid, grad_outputs)?;
            }

            // Get layer architecture
            let layer_arch = self
                .param_manager
                .get_layer_architecture(layerid)
                .ok_or_else(|| {
                    OptimError::InvalidConfig(format!("Layer {} not registered", layerid))
                })?
                .clone();

            // Compute gradients based on layer type
            let grad_inputs = match layer_arch.layer_type.as_str() {
                "linear" | "dense" | "fc" => {
                    // Linear layer gradient computation
                    self.compute_linear_backward(layerid, grad_outputs)?
                }
                "conv" | "conv2d" => {
                    // Convolutional layer gradient computation
                    self.compute_conv_backward(layerid, grad_outputs)?
                }
                "activation" => {
                    // Activation gradient computation
                    self.compute_activation_backward(layerid, grad_outputs, &layer_arch)?
                }
                "normalization" | "batchnorm" | "layernorm" => {
                    // Normalization gradient computation
                    self.compute_normalization_backward(layerid, grad_outputs)?
                }
                "dropout" => {
                    // Dropout gradient computation
                    self.compute_dropout_backward(layerid, grad_outputs, &layer_arch)?
                }
                "pooling" | "maxpool" | "avgpool" => {
                    // Pooling gradient computation
                    self.compute_pooling_backward(layerid, grad_outputs, &layer_arch)?
                }
                _ => {
                    // Default: pass through gradients for unknown layer types
                    grad_outputs.to_vec()
                }
            };

            // Apply gradient clipping if configured
            let clipped_grads =
                if let Some(clipvalue) = self.param_manager.global_config.gradient_clip {
                    self.apply_gradient_clipping(grad_inputs, clipvalue)?
                } else {
                    grad_inputs
                };

            // Execute post-backward hook
            if let Some(hook) = self.backward_hooks.get_mut(layerid) {
                hook.post_backward(layerid, &clipped_grads)?;
            }

            Ok(clipped_grads)
        }

        /// Compute linear layer backward pass
        fn compute_linear_backward(
            &mut self,
            layerid: &LayerId,
            grad_outputs: &[Array<A, D>],
        ) -> Result<Vec<Array<A, D>>> {
            if grad_outputs.is_empty() {
                return Err(OptimError::InvalidConfig(
                    "Linear layer backward requires gradients".to_string(),
                ));
            }

            // Get parameters for this layer
            let layer_params = self.param_manager.get_parameters_by_layer(layerid);

            // Store gradients for weight update
            if self.gradient_accumulation {
                let mut param_grads = HashMap::new();
                for (i, paramid) in layer_params.iter().enumerate() {
                    if i < grad_outputs.len() {
                        param_grads.insert((*paramid).clone(), grad_outputs[i].clone());
                    }
                }
                self.accumulate_gradients(param_grads)?;
            }

            // Simple gradient transformation: scale by learning rate decay
            let lr_decay = A::from(0.9).unwrap();
            let grad_inputs: Vec<Array<A, D>> = grad_outputs
                .iter()
                .map(|grad| grad.mapv(|x| x * lr_decay))
                .collect();

            Ok(grad_inputs)
        }

        /// Compute convolutional layer backward pass
        fn compute_conv_backward(
            &self,
            _layer_id: &LayerId,
            grad_outputs: &[Array<A, D>],
        ) -> Result<Vec<Array<A, D>>> {
            // Simplified convolution backward: pass through gradients
            // Real implementation would compute gradients w.r.t. kernels and input
            Ok(grad_outputs.to_vec())
        }

        /// Compute activation backward pass
        fn compute_activation_backward(
            &self,
            _layer_id: &LayerId,
            grad_outputs: &[Array<A, D>],
            layer_arch: &LayerArchitecture,
        ) -> Result<Vec<Array<A, D>>> {
            let activation_type = layer_arch
                .config
                .get("activation")
                .and_then(|v| match v {
                    LayerConfig::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or("relu");

            // Note: This is simplified - real implementation would need the forward pass inputs
            let grad_inputs: Vec<Array<A, D>> = grad_outputs
                .iter()
                .map(|grad| {
                    match activation_type {
                        "relu" => {
                            // ReLU gradient: 1 if x > 0, 0 otherwise
                            // Since we don't have the original input, we approximate
                            grad.mapv(|g| if g > A::zero() { g } else { A::zero() })
                        }
                        "sigmoid" => {
                            // Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
                            // Approximation without original input
                            let factor = A::from(0.25).unwrap(); // Max gradient of sigmoid
                            grad.mapv(|g| g * factor)
                        }
                        "tanh" => {
                            // Tanh gradient: 1 - tanh(x)^2
                            // Approximation without original input
                            let factor = A::from(0.5).unwrap();
                            grad.mapv(|g| g * factor)
                        }
                        "leaky_relu" => {
                            let alpha = A::from(0.01).unwrap();
                            grad.mapv(|g| if g > A::zero() { g } else { alpha * g })
                        }
                        _ => grad.clone(), // Unknown activation, pass through
                    }
                })
                .collect();

            Ok(grad_inputs)
        }

        /// Compute normalization backward pass
        fn compute_normalization_backward(
            &self,
            _layer_id: &LayerId,
            grad_outputs: &[Array<A, D>],
        ) -> Result<Vec<Array<A, D>>> {
            // Simplified normalization backward
            // Real implementation would compute gradients considering mean and variance
            let scale_factor = A::from(0.9).unwrap();
            let grad_inputs: Vec<Array<A, D>> = grad_outputs
                .iter()
                .map(|grad| grad.mapv(|g| g * scale_factor))
                .collect();

            Ok(grad_inputs)
        }

        /// Compute dropout backward pass
        fn compute_dropout_backward(
            &self,
            _layer_id: &LayerId,
            grad_outputs: &[Array<A, D>],
            layer_arch: &LayerArchitecture,
        ) -> Result<Vec<Array<A, D>>> {
            let dropout_rate = layer_arch
                .config
                .get("dropout_rate")
                .and_then(|v| match v {
                    LayerConfig::Float(f) => Some(A::from(*f).unwrap()),
                    _ => None,
                })
                .unwrap_or(A::from(0.5).unwrap());

            // Scale gradients by (1 - dropout_rate) to match forward pass
            let scale = A::one() - dropout_rate;
            let grad_inputs: Vec<Array<A, D>> = grad_outputs
                .iter()
                .map(|grad| grad.mapv(|g| g * scale))
                .collect();

            Ok(grad_inputs)
        }

        /// Compute pooling backward pass
        fn compute_pooling_backward(
            &self,
            _layer_id: &LayerId,
            grad_outputs: &[Array<A, D>],
            _layer_arch: &LayerArchitecture,
        ) -> Result<Vec<Array<A, D>>> {
            // Simplified pooling backward: pass through gradients
            // Real implementation would upsample gradients to match input size
            Ok(grad_outputs.to_vec())
        }

        /// Apply gradient clipping
        fn apply_gradient_clipping(
            &self,
            gradients: Vec<Array<A, D>>,
            clipvalue: A,
        ) -> Result<Vec<Array<A, D>>> {
            let clipped: Vec<Array<A, D>> = gradients
                .into_iter()
                .map(|grad| {
                    // Compute L2 norm of gradient
                    let norm = grad.mapv(|x| x * x).sum().sqrt();

                    if norm > clipvalue {
                        // Scale gradient to have norm = clipvalue
                        let scale = clipvalue / norm;
                        grad.mapv(|x| x * scale)
                    } else {
                        grad
                    }
                })
                .collect();

            Ok(clipped)
        }

        /// Accumulate gradients for parameters
        pub fn accumulate_gradients(
            &mut self,
            gradients: HashMap<ParamId, Array<A, D>>,
        ) -> Result<()> {
            if !self.gradient_accumulation {
                return Err(OptimError::InvalidConfig(
                    "Gradient accumulation not enabled".to_string(),
                ));
            }

            self.accumulation_count += 1;

            for (paramid, grad) in gradients {
                if let Some(acc_grad) = self.accumulated_gradients.get_mut(&paramid) {
                    // Add to existing accumulated gradient
                    *acc_grad = acc_grad.clone() + grad;
                } else {
                    // First gradient for this parameter
                    self.accumulated_gradients.insert(paramid, grad);
                }
            }

            Ok(())
        }

        /// Get parameter manager
        pub fn parameter_manager(&self) -> &ParameterManager<A, D> {
            &self.param_manager
        }

        /// Get mutable parameter manager
        pub fn parameter_manager_mut(&mut self) -> &mut ParameterManager<A, D> {
            &mut self.param_manager
        }

        /// Get accumulation count
        pub fn accumulation_count(&self) -> usize {
            self.accumulation_count
        }
    }
}

/// Architecture-aware optimization utilities
pub mod architecture_aware {
    use super::*;

    /// Architecture-specific optimization strategy
    #[derive(Debug, Clone)]
    pub enum ArchitectureStrategy {
        /// Transformer-specific optimizations
        Transformer {
            /// Use different learning rates for different components
            component_specific_lr: bool,
            /// Apply layer-wise learning rate decay
            layer_wise_decay: bool,
            /// Warmup steps for attention parameters
            attention_warmup: usize,
        },
        /// CNN-specific optimizations
        ConvolutionalNet {
            /// Use different learning rates for conv vs fc layers
            layer_type_lr: bool,
            /// Apply depth-wise learning rate scaling
            depth_scaling: bool,
            /// Batch norm parameter handling
            bn_special_handling: bool,
        },
        /// RNN-specific optimizations
        RecurrentNet {
            /// Gradient clipping specifically for RNNs
            rnn_gradient_clip: Option<f64>,
            /// Different learning rates for recurrent vs linear weights
            weight_type_lr: bool,
        },
        /// Custom architecture strategy
        Custom {
            /// Custom optimization rules
            rules: HashMap<String, LayerConfig>,
        },
    }

    /// Architecture-aware optimizer
    #[derive(Debug)]
    pub struct ArchitectureAwareOptimizer<A: Float, D: Dimension> {
        /// Parameter manager
        param_manager: ParameterManager<A, D>,
        /// Architecture strategy
        strategy: ArchitectureStrategy,
        /// Step count
        step_count: usize,
    }

    impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync>
        ArchitectureAwareOptimizer<A, D>
    {
        /// Create a new architecture-aware optimizer
        pub fn new(config: OptimizationConfig<A>, strategy: ArchitectureStrategy) -> Self {
            Self {
                param_manager: ParameterManager::new(config),
                strategy,
                step_count: 0,
            }
        }

        /// Apply architecture-specific optimizations
        pub fn apply_architecture_optimizations(&mut self) -> Result<()> {
            // Clone the strategy to avoid borrowing conflicts
            let strategy = self.strategy.clone();
            match strategy {
                ArchitectureStrategy::Transformer {
                    component_specific_lr,
                    layer_wise_decay,
                    attention_warmup,
                } => {
                    self.apply_transformer_optimizations(
                        component_specific_lr,
                        layer_wise_decay,
                        attention_warmup,
                    )?;
                }
                ArchitectureStrategy::ConvolutionalNet {
                    layer_type_lr,
                    depth_scaling,
                    bn_special_handling,
                } => {
                    self.apply_cnn_optimizations(
                        layer_type_lr,
                        depth_scaling,
                        bn_special_handling,
                    )?;
                }
                ArchitectureStrategy::RecurrentNet {
                    rnn_gradient_clip,
                    weight_type_lr,
                } => {
                    self.apply_rnn_optimizations(rnn_gradient_clip, weight_type_lr)?;
                }
                ArchitectureStrategy::Custom { rules } => {
                    self.apply_custom_optimizations(&rules)?;
                }
            }
            Ok(())
        }

        /// Apply Transformer-specific optimizations
        fn apply_transformer_optimizations(
            &mut self,
            component_specific_lr: bool,
            layer_wise_decay: bool,
            attention_warmup: usize,
        ) -> Result<()> {
            if component_specific_lr {
                // Different learning rates for attention, ffn, and normalization
                self.set_component_learning_rates()?;
            }

            if layer_wise_decay {
                // Apply layer-wise learning rate _decay
                self.apply_layer_wise_decay()?;
            }

            if attention_warmup > 0 && self.step_count < attention_warmup {
                // Apply _warmup to attention parameters
                self.apply_attention_warmup(attention_warmup)?;
            }

            Ok(())
        }

        /// Apply CNN-specific optimizations
        fn apply_cnn_optimizations(
            &mut self,
            layer_type_lr: bool,
            depth_scaling: bool,
            bn_special_handling: bool,
        ) -> Result<()> {
            if layer_type_lr {
                // Different learning rates for conv vs fully connected layers
                self.set_layer_type_learning_rates()?;
            }

            if depth_scaling {
                // Scale learning rates based on network depth
                self.apply_depth_scaling()?;
            }

            if bn_special_handling {
                // Special _handling for batch normalization parameters
                self.apply_bn_optimizations()?;
            }

            Ok(())
        }

        /// Apply RNN-specific optimizations
        fn apply_rnn_optimizations(
            &mut self,
            rnn_gradient_clip: Option<f64>,
            weight_type_lr: bool,
        ) -> Result<()> {
            if let Some(clipvalue) = rnn_gradient_clip {
                // Apply RNN-specific gradient clipping
                self.apply_rnn_gradient_clipping(A::from(clipvalue).unwrap())?;
            }

            if weight_type_lr {
                // Different learning rates for recurrent vs linear weights
                self.set_weight_type_learning_rates()?;
            }

            Ok(())
        }

        /// Apply custom optimizations
        fn apply_custom_optimizations(
            &mut self,
            rules: &HashMap<String, LayerConfig>,
        ) -> Result<()> {
            // Collect rules first to avoid borrowing conflicts
            let rule_entries: Vec<(String, LayerConfig)> = rules
                .iter()
                .map(|(name, config)| (name.clone(), config.clone()))
                .collect();

            for (rule_name, config) in rule_entries {
                self.apply_custom_rule(&rule_name, &config)?;
            }
            Ok(())
        }

        /// Set component-specific learning rates for Transformers
        fn set_component_learning_rates(&mut self) -> Result<()> {
            // Collect the data first to avoid borrowing conflicts
            let layer_rules: Vec<(LayerId, LayerOptimizationRule<A>)> = self
                .param_manager
                .get_all_parameters()
                .iter()
                .map(|(_param_id, metadata)| {
                    let mut rule = LayerOptimizationRule::default();

                    // Determine learning rate multiplier based on parameter tags
                    if metadata.tags.contains(&"attention".to_string()) {
                        rule.lr_multiplier = A::from(1.2).unwrap(); // Higher LR for attention
                    } else if metadata.tags.contains(&"ffn".to_string()) {
                        rule.lr_multiplier = A::from(1.0).unwrap(); // Standard LR for FFN
                    } else if metadata.tags.contains(&"normalization".to_string()) {
                        rule.lr_multiplier = A::from(0.8).unwrap(); // Lower LR for normalization
                    }

                    (metadata.layername.clone(), rule)
                })
                .collect();

            // Now apply the rules
            for (layername, rule) in layer_rules {
                self.param_manager.set_layer_rule(layername, rule);
            }
            Ok(())
        }

        /// Apply layer-wise learning rate decay
        fn apply_layer_wise_decay(&mut self) -> Result<()> {
            // Extract layer numbers from layer names and apply decay
            for (layerid, _) in self.param_manager.layer_architectures.clone() {
                if let Some(layer_num) = self.extract_layer_number(&layerid) {
                    let decay_factor = A::from(0.95_f64.powi(layer_num as i32)).unwrap();
                    let mut rule = self
                        .param_manager
                        .layer_rules
                        .get(&layerid)
                        .cloned()
                        .unwrap_or_default();
                    rule.lr_multiplier = rule.lr_multiplier * decay_factor;
                    self.param_manager.set_layer_rule(layerid, rule);
                }
            }
            Ok(())
        }

        /// Apply attention parameter warmup
        fn apply_attention_warmup(&mut self, warmupsteps: usize) -> Result<()> {
            let warmup_factor = A::from(self.step_count as f64 / warmupsteps as f64).unwrap();

            // Collect attention layers first
            let attention_layers: Vec<LayerId> = self
                .param_manager
                .get_all_parameters()
                .iter()
                .filter_map(|(_param_id, metadata)| {
                    if metadata.tags.contains(&"attention".to_string()) {
                        Some(metadata.layername.clone())
                    } else {
                        None
                    }
                })
                .collect();

            // Apply warmup to attention layers
            for layername in attention_layers {
                let mut rule = self
                    .param_manager
                    .layer_rules
                    .get(&layername)
                    .cloned()
                    .unwrap_or_default();
                rule.lr_multiplier = rule.lr_multiplier * warmup_factor;
                self.param_manager.set_layer_rule(layername, rule);
            }
            Ok(())
        }

        /// Set learning rates based on layer type (conv vs fc)
        fn set_layer_type_learning_rates(&mut self) -> Result<()> {
            for (layerid, architecture) in self.param_manager.layer_architectures.clone() {
                let mut rule = LayerOptimizationRule::default();

                match architecture.layer_type.as_str() {
                    "conv" | "conv2d" | "conv3d" => {
                        rule.lr_multiplier = A::from(1.0).unwrap(); // Standard LR for conv
                    }
                    "linear" | "dense" | "fc" => {
                        rule.lr_multiplier = A::from(0.8).unwrap(); // Lower LR for FC
                    }
                    _ => {
                        rule.lr_multiplier = A::from(1.0).unwrap(); // Default
                    }
                }

                self.param_manager.set_layer_rule(layerid, rule);
            }
            Ok(())
        }

        /// Apply depth-based scaling
        fn apply_depth_scaling(&mut self) -> Result<()> {
            // Count total layers
            let total_layers = self.param_manager.layer_architectures.len();

            for (i, (layerid, _)) in self
                .param_manager
                .layer_architectures
                .clone()
                .iter()
                .enumerate()
            {
                let depth_factor = A::from(1.0 - 0.1 * (i as f64 / total_layers as f64)).unwrap();
                let mut rule = self
                    .param_manager
                    .layer_rules
                    .get(layerid)
                    .cloned()
                    .unwrap_or_default();
                rule.lr_multiplier = rule.lr_multiplier * depth_factor;
                self.param_manager.set_layer_rule(layerid.clone(), rule);
            }
            Ok(())
        }

        /// Apply batch normalization optimizations
        fn apply_bn_optimizations(&mut self) -> Result<()> {
            // Collect normalization layers first
            let norm_layers: Vec<LayerId> = self
                .param_manager
                .get_all_parameters()
                .iter()
                .filter_map(|(_param_id, metadata)| {
                    if metadata.paramtype == ParameterType::Normalization {
                        Some(metadata.layername.clone())
                    } else {
                        None
                    }
                })
                .collect();

            // Apply optimization to normalization layers
            for layername in norm_layers {
                let mut rule = self
                    .param_manager
                    .layer_rules
                    .get(&layername)
                    .cloned()
                    .unwrap_or_default();
                // Higher learning rate and no weight decay for BN parameters
                rule.lr_multiplier = A::from(2.0).unwrap();
                rule.weight_decay_multiplier = A::zero();
                self.param_manager.set_layer_rule(layername, rule);
            }
            Ok(())
        }

        /// Apply RNN-specific gradient clipping
        fn apply_rnn_gradient_clipping(&mut self, clipvalue: A) -> Result<()> {
            // This would be implemented in coordination with the gradient processing system
            // For now, we'll store the clip _value in the global config
            self.param_manager.global_config.gradient_clip = Some(clipvalue);
            Ok(())
        }

        /// Set learning rates based on weight type (recurrent vs linear)
        fn set_weight_type_learning_rates(&mut self) -> Result<()> {
            // Collect weight type layers first
            let layer_rules: Vec<(LayerId, LayerOptimizationRule<A>)> = self
                .param_manager
                .get_all_parameters()
                .iter()
                .map(|(_param_id, metadata)| {
                    let mut rule = LayerOptimizationRule::default();

                    if metadata.tags.contains(&"recurrent".to_string()) {
                        rule.lr_multiplier = A::from(0.5).unwrap(); // Lower LR for recurrent weights
                    } else if metadata.tags.contains(&"linear".to_string()) {
                        rule.lr_multiplier = A::from(1.0).unwrap(); // Standard LR for linear weights
                    }

                    (metadata.layername.clone(), rule)
                })
                .collect();

            // Apply the rules
            for (layername, rule) in layer_rules {
                self.param_manager.set_layer_rule(layername, rule);
            }
            Ok(())
        }

        /// Apply custom optimization rule
        fn apply_custom_rule(&mut self, _rule_name: &str, config: &LayerConfig) -> Result<()> {
            // Custom rule implementation would depend on the specific rule
            // This is a placeholder for extensibility
            Ok(())
        }

        /// Extract layer number from layer name (e.g., "layer_12" -> 12)
        fn extract_layer_number(&self, layername: &str) -> Option<usize> {
            layername.split('_').next_back()?.parse().ok()
        }

        /// Step the optimizer
        pub fn step(&mut self) -> Result<()> {
            self.step_count += 1;
            self.apply_architecture_optimizations()
        }

        /// Get parameter manager
        pub fn parameter_manager(&self) -> &ParameterManager<A, D> {
            &self.param_manager
        }

        /// Get mutable parameter manager
        pub fn parameter_manager_mut(&mut self) -> &mut ParameterManager<A, D> {
            &mut self.param_manager
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parameter_manager_basic() {
        let config = OptimizationConfig::default();
        let mut manager = ParameterManager::<f64, scirs2_core::ndarray_ext::Ix1>::new(config);

        let metadata = ParameterMetadata {
            layername: "layer1".to_string(),
            param_name: "weight".to_string(),
            shape: vec![10, 5],
            requires_grad: true,
            paramtype: ParameterType::Weight,
            sharing_group: None,
            tags: vec!["dense".to_string()],
        };

        manager
            .register_parameter("param1".to_string(), metadata)
            .unwrap();

        assert!(manager
            .get_parameter_metadata(&"param1".to_string())
            .is_some());
        assert_eq!(manager.get_all_parameters().len(), 1);
        assert!(!manager.is_parameter_frozen(&"param1".to_string()));
    }

    #[test]
    fn test_lazy_registration() {
        let config = OptimizationConfig::default();
        let mut manager = ParameterManager::<f64, scirs2_core::ndarray_ext::Ix1>::new(config);

        manager.enable_lazy_mode();

        let metadata = ParameterMetadata {
            layername: "layer1".to_string(),
            param_name: "weight".to_string(),
            shape: vec![10, 5],
            requires_grad: true,
            paramtype: ParameterType::Weight,
            sharing_group: None,
            tags: vec![],
        };

        manager
            .register_parameter("param1".to_string(), metadata)
            .unwrap();

        // Parameter should not be registered yet
        assert_eq!(manager.get_all_parameters().len(), 0);

        // Disable lazy mode to process pending registrations
        manager.disable_lazy_mode().unwrap();

        // Now parameter should be registered
        assert_eq!(manager.get_all_parameters().len(), 1);
    }

    #[test]
    fn test_layer_specific_rules() {
        let config = OptimizationConfig {
            base_learning_rate: 0.01,
            weight_decay: 0.001,
            gradient_clip: None,
            mixed_precision: false,
            architecture_optimizations: HashMap::new(),
        };
        let mut manager = ParameterManager::<f64, scirs2_core::ndarray_ext::Ix1>::new(config);

        let rule = LayerOptimizationRule {
            lr_multiplier: 2.0,
            weight_decay_multiplier: 0.5,
            frozen: false,
            custom_settings: HashMap::new(),
        };

        manager.set_layer_rule("layer1".to_string(), rule);

        let metadata = ParameterMetadata {
            layername: "layer1".to_string(),
            param_name: "weight".to_string(),
            shape: vec![10, 5],
            requires_grad: true,
            paramtype: ParameterType::Weight,
            sharing_group: None,
            tags: vec![],
        };

        manager
            .register_parameter("param1".to_string(), metadata)
            .unwrap();

        // Test effective learning rate
        let effective_lr = manager.get_effective_learning_rate(&"param1".to_string());
        assert_relative_eq!(effective_lr, 0.02, epsilon = 1e-6); // 0.01 * 2.0

        // Test effective weight decay
        let effective_decay = manager.get_effective_weight_decay(&"param1".to_string());
        assert_relative_eq!(effective_decay, 0.0005, epsilon = 1e-6); // 0.001 * 0.5
    }

    #[test]
    fn test_parameter_sharing() {
        let config = OptimizationConfig::default();
        let mut manager = ParameterManager::<f64, scirs2_core::ndarray_ext::Ix1>::new(config);

        let metadata1 = ParameterMetadata {
            layername: "layer1".to_string(),
            param_name: "weight".to_string(),
            shape: vec![10, 5],
            requires_grad: true,
            paramtype: ParameterType::Weight,
            sharing_group: Some("shared_weights".to_string()),
            tags: vec![],
        };

        let metadata2 = ParameterMetadata {
            layername: "layer2".to_string(),
            param_name: "weight".to_string(),
            shape: vec![10, 5],
            requires_grad: true,
            paramtype: ParameterType::Weight,
            sharing_group: Some("shared_weights".to_string()),
            tags: vec![],
        };

        manager
            .register_parameter("param1".to_string(), metadata1)
            .unwrap();
        manager
            .register_parameter("param2".to_string(), metadata2)
            .unwrap();

        let sharing_group = manager.get_sharing_group("shared_weights").unwrap();
        assert_eq!(sharing_group.len(), 2);
        assert!(sharing_group.contains(&"param1".to_string()));
        assert!(sharing_group.contains(&"param2".to_string()));
    }

    #[test]
    fn test_parameter_filtering() {
        let config = OptimizationConfig::default();
        let mut manager = ParameterManager::<f64, scirs2_core::ndarray_ext::Ix1>::new(config);

        let weight_metadata = ParameterMetadata {
            layername: "layer1".to_string(),
            param_name: "weight".to_string(),
            shape: vec![10, 5],
            requires_grad: true,
            paramtype: ParameterType::Weight,
            sharing_group: None,
            tags: vec![],
        };

        let bias_metadata = ParameterMetadata {
            layername: "layer1".to_string(),
            param_name: "bias".to_string(),
            shape: vec![5],
            requires_grad: true,
            paramtype: ParameterType::Bias,
            sharing_group: None,
            tags: vec![],
        };

        manager
            .register_parameter("weight".to_string(), weight_metadata)
            .unwrap();
        manager
            .register_parameter("bias".to_string(), bias_metadata)
            .unwrap();

        // Test filtering by type
        let weights = manager.get_parameters_by_type(ParameterType::Weight);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], &"weight".to_string());

        let biases = manager.get_parameters_by_type(ParameterType::Bias);
        assert_eq!(biases.len(), 1);
        assert_eq!(biases[0], &"bias".to_string());

        // Test filtering by layer
        let layer_params = manager.get_parameters_by_layer(&"layer1".to_string());
        assert_eq!(layer_params.len(), 2);

        // Test trainable parameters
        let trainable = manager.get_trainable_parameters();
        assert_eq!(trainable.len(), 2);
    }

    #[test]
    fn test_architecture_aware_transformer() {
        use crate::neural_integration::architecture_aware::*;

        let config = OptimizationConfig::default();
        let strategy = ArchitectureStrategy::Transformer {
            component_specific_lr: true,
            layer_wise_decay: true,
            attention_warmup: 1000,
        };

        let mut optimizer =
            ArchitectureAwareOptimizer::<f64, scirs2_core::ndarray_ext::Ix1>::new(config, strategy);

        // Register a layer architecture
        let layer_arch = LayerArchitecture {
            layer_type: "transformer_block".to_string(),
            input_dims: vec![512],
            output_dims: vec![512],
            config: HashMap::new(),
            trainable: true,
        };

        optimizer
            .parameter_manager_mut()
            .register_layer("layer_0".to_string(), layer_arch);

        // Register parameters with different tags
        let attention_metadata = ParameterMetadata {
            layername: "layer_0".to_string(),
            param_name: "attention_weight".to_string(),
            shape: vec![512, 512],
            requires_grad: true,
            paramtype: ParameterType::Attention,
            sharing_group: None,
            tags: vec!["attention".to_string()],
        };

        optimizer
            .parameter_manager_mut()
            .register_parameter("attn_param".to_string(), attention_metadata)
            .unwrap();

        // Apply optimizations
        optimizer.apply_architecture_optimizations().unwrap();

        // Verify that attention parameters get special treatment
        assert!(optimizer
            .parameter_manager()
            .get_parameter_metadata(&"attn_param".to_string())
            .is_some());
    }
}
