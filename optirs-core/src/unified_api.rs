// Unified API consistent with popular deep learning frameworks
//
// This module provides a unified interface that closely follows the design patterns
// of popular deep learning frameworks like PyTorch, TensorFlow, and JAX/Optax.
//
// # Design Principles
//
// - **Parameter Groups**: Support for different optimization parameters for different layers
// - **State Management**: Automatic handling of optimizer state
// - **Framework Consistency**: APIs that feel familiar to PyTorch/TensorFlow users
// - **Flexible Configuration**: Easy-to-use builder patterns
// - **Scheduler Integration**: Seamless integration with learning rate schedulers

use crate::error::{OptimError, Result};
use crate::schedulers::LearningRateScheduler;
use scirs2_core::ndarray::{Array, Array1, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

/// Unified optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig<A: Float> {
    /// Learning rate
    pub lr: A,
    /// Weight decay (L2 regularization)
    pub weight_decay: A,
    /// Gradient clipping value (optional)
    pub grad_clip: Option<A>,
    /// Additional optimizer-specific parameters
    pub params: HashMap<String, A>,
}

impl<A: Float + Send + Sync> Default for OptimizerConfig<A> {
    fn default() -> Self {
        Self {
            lr: A::from(0.001).expect("unwrap failed"),
            weight_decay: A::zero(),
            grad_clip: None,
            params: HashMap::new(),
        }
    }
}

impl<A: Float + Send + Sync> OptimizerConfig<A> {
    /// Create a new optimizer configuration with the given learning rate
    pub fn new(lr: A) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    /// Set weight decay
    pub fn weight_decay(mut self, weightdecay: A) -> Self {
        self.weight_decay = weightdecay;
        self
    }

    /// Set gradient clipping
    pub fn grad_clip(mut self, gradclip: A) -> Self {
        self.grad_clip = Some(gradclip);
        self
    }

    /// Add a custom parameter
    pub fn param<S: Into<String>>(mut self, key: S, value: A) -> Self {
        self.params.insert(key.into(), value);
        self
    }

    /// Set multiple parameters at once
    pub fn params(mut self, params: HashMap<String, A>) -> Self {
        self.params.extend(params);
        self
    }
}

/// Parameter tensor wrapper for unified API
#[derive(Debug, Clone)]
pub struct Parameter<A: Float, D: Dimension> {
    /// Parameter data
    pub data: Array<A, D>,
    /// Gradient data (optional)
    pub grad: Option<Array<A, D>>,
    /// Whether this parameter requires gradients
    pub requires_grad: bool,
    /// Parameter name/identifier
    pub name: String,
}

impl<A: Float + ScalarOperand, D: Dimension + Send + Sync> Parameter<A, D> {
    /// Create a new parameter
    pub fn new<S: Into<String>>(data: Array<A, D>, name: S) -> Self {
        Self {
            data,
            grad: None,
            requires_grad: true,
            name: name.into(),
        }
    }

    /// Create a parameter that doesn't require gradients
    pub fn no_grad<S: Into<String>>(data: Array<A, D>, name: S) -> Self {
        Self {
            data,
            grad: None,
            requires_grad: false,
            name: name.into(),
        }
    }

    /// Set gradient for this parameter
    pub fn set_grad(&mut self, grad: Array<A, D>) {
        if self.requires_grad {
            self.grad = Some(grad);
        }
    }

    /// Clear gradients
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Get gradient reference
    pub fn grad(&self) -> Option<&Array<A, D>> {
        self.grad.as_ref()
    }

    /// Apply gradient clipping if specified
    pub fn clip_grad(&mut self, maxnorm: A) -> Result<()> {
        if let Some(ref mut grad) = self.grad {
            let _norm = grad
                .iter()
                .map(|x| (*x) * (*x))
                .fold(A::zero(), |acc, x| acc + x)
                .sqrt();
            if _norm > maxnorm {
                let scale = maxnorm / _norm;
                grad.mapv_inplace(|x| x * scale);
            }
        }
        Ok(())
    }
}

/// Unified optimizer interface
pub trait UnifiedOptimizer<A: Float> {
    /// Get optimizer configuration
    fn config(&self) -> &OptimizerConfig<A>;

    /// Update a single parameter
    fn step_param<D: Dimension>(&mut self, param: &mut Parameter<A, D>) -> Result<()>
    where
        A: ScalarOperand + Debug;

    /// Update multiple parameters
    fn step_params<D: Dimension>(&mut self, params: &mut [Parameter<A, D>]) -> Result<()>
    where
        A: ScalarOperand + Debug,
    {
        for param in params.iter_mut() {
            self.step_param(param)?;
        }
        Ok(())
    }

    /// Zero gradients for all parameters
    fn zero_grad<D: Dimension>(&self, params: &mut [Parameter<A, D>]) {
        for param in params.iter_mut() {
            param.grad = None;
        }
    }

    /// Update learning rate
    fn set_lr(&mut self, lr: A);

    /// Get current learning rate
    fn get_lr(&self) -> A;

    /// Serialize the full optimizer state (configuration + per-parameter buffers)
    ///
    /// Values are stored as little-endian `f64` blobs, so the format is dependency
    /// free and stable across `f32` / `f64` optimizers. Keys are namespaced:
    ///
    /// | key | contents |
    /// |-----|----------|
    /// | `format.version` | one `f64`, currently `1` |
    /// | `config.lr` / `config.weight_decay` | one `f64` each |
    /// | `config.grad_clip` | one `f64`, absent when clipping is disabled |
    /// | `config.param.<name>` | one `f64` per optimizer-specific hyperparameter |
    /// | `<buffer>.<parameter name>` | the buffer contents, one `f64` per element |
    fn state_dict(&self) -> Result<HashMap<String, Vec<u8>>>;

    /// Restore state previously produced by [`UnifiedOptimizer::state_dict`]
    ///
    /// # Errors
    ///
    /// Returns an error when the payload is truncated (not a whole number of `f64`
    /// values), when the format version is unknown, or when a restored buffer does
    /// not match the shape of the buffer it replaces.
    fn load_state_dict(&mut self, statedict: HashMap<String, Vec<u8>>) -> Result<()>;
}

/// Serialization format version written into every state dictionary
const STATE_DICT_VERSION: f64 = 1.0;

/// Key holding the state-dictionary format version
const KEY_FORMAT_VERSION: &str = "format.version";

/// Encodes `f64` values as a little-endian byte blob
fn encode_f64_slice(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Decodes a little-endian byte blob back into `f64` values
fn decode_f64_slice(key: &str, bytes: &[u8]) -> Result<Vec<f64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(OptimError::InvalidConfig(format!(
            "state dict entry '{}' is truncated: {} bytes is not a multiple of 8",
            key,
            bytes.len()
        )));
    }

    let mut values = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        values.push(f64::from_le_bytes(buf));
    }
    Ok(values)
}

/// Converts a floating-point value into the state-dict representation
fn to_state_value<A: Float>(key: &str, value: A) -> Result<f64> {
    value.to_f64().ok_or_else(|| {
        OptimError::InvalidConfig(format!("state dict entry '{}' is not representable", key))
    })
}

/// Converts a state-dict value back into the optimizer's floating-point type
fn from_state_value<A: Float>(key: &str, value: f64) -> Result<A> {
    A::from(value).ok_or_else(|| {
        OptimError::InvalidConfig(format!(
            "state dict entry '{}' holds a value that is not representable in the target type",
            key
        ))
    })
}

/// Reads exactly one scalar out of a state-dict entry
fn decode_scalar(key: &str, bytes: &[u8]) -> Result<f64> {
    let values = decode_f64_slice(key, bytes)?;
    match values.as_slice() {
        [single] => Ok(*single),
        other => Err(OptimError::InvalidConfig(format!(
            "state dict entry '{}' must hold exactly one value, found {}",
            key,
            other.len()
        ))),
    }
}

/// Serializes the shared [`OptimizerConfig`] portion of a state dictionary
fn encode_config<A: Float>(
    config: &OptimizerConfig<A>,
    target: &mut HashMap<String, Vec<u8>>,
) -> Result<()> {
    target.insert(
        KEY_FORMAT_VERSION.to_string(),
        encode_f64_slice(&[STATE_DICT_VERSION]),
    );
    target.insert(
        "config.lr".to_string(),
        encode_f64_slice(&[to_state_value("config.lr", config.lr)?]),
    );
    target.insert(
        "config.weight_decay".to_string(),
        encode_f64_slice(&[to_state_value("config.weight_decay", config.weight_decay)?]),
    );
    if let Some(clip) = config.grad_clip {
        target.insert(
            "config.grad_clip".to_string(),
            encode_f64_slice(&[to_state_value("config.grad_clip", clip)?]),
        );
    }
    for (name, value) in config.params.iter() {
        let key = format!("config.param.{}", name);
        let encoded = encode_f64_slice(&[to_state_value(&key, *value)?]);
        target.insert(key, encoded);
    }
    Ok(())
}

/// Restores the shared [`OptimizerConfig`] portion of a state dictionary
fn decode_config<A: Float + Send + Sync>(
    state: &HashMap<String, Vec<u8>>,
    config: &mut OptimizerConfig<A>,
) -> Result<()> {
    let version_bytes = state.get(KEY_FORMAT_VERSION).ok_or_else(|| {
        OptimError::InvalidConfig(format!("state dict is missing '{}'", KEY_FORMAT_VERSION))
    })?;
    let version = decode_scalar(KEY_FORMAT_VERSION, version_bytes)?;
    if version != STATE_DICT_VERSION {
        return Err(OptimError::InvalidConfig(format!(
            "unsupported state dict version {} (expected {})",
            version, STATE_DICT_VERSION
        )));
    }

    if let Some(bytes) = state.get("config.lr") {
        config.lr = from_state_value("config.lr", decode_scalar("config.lr", bytes)?)?;
    }
    if let Some(bytes) = state.get("config.weight_decay") {
        config.weight_decay = from_state_value(
            "config.weight_decay",
            decode_scalar("config.weight_decay", bytes)?,
        )?;
    }
    config.grad_clip = match state.get("config.grad_clip") {
        Some(bytes) => Some(from_state_value(
            "config.grad_clip",
            decode_scalar("config.grad_clip", bytes)?,
        )?),
        None => None,
    };

    for (key, bytes) in state.iter() {
        if let Some(name) = key.strip_prefix("config.param.") {
            let value = from_state_value(key, decode_scalar(key, bytes)?)?;
            config.params.insert(name.to_string(), value);
        }
    }
    Ok(())
}

/// Restores a named collection of `Array1` buffers, validating their shapes
fn decode_buffers<A: Float>(
    state: &HashMap<String, Vec<u8>>,
    prefix: &str,
    existing: &HashMap<String, Array1<A>>,
) -> Result<HashMap<String, Array1<A>>> {
    let mut restored = HashMap::new();
    for (key, bytes) in state.iter() {
        let name = match key.strip_prefix(prefix) {
            Some(name) => name,
            None => continue,
        };
        let values = decode_f64_slice(key, bytes)?;

        if let Some(current) = existing.get(name) {
            if current.len() != values.len() {
                return Err(OptimError::DimensionMismatch(format!(
                    "state dict buffer '{}' has {} elements but the optimizer holds {}",
                    key,
                    values.len(),
                    current.len()
                )));
            }
        }

        let mut buffer = Array1::zeros(values.len());
        for (slot, value) in buffer.iter_mut().zip(values.iter()) {
            *slot = from_state_value(key, *value)?;
        }
        restored.insert(name.to_string(), buffer);
    }
    Ok(restored)
}

/// Serializes a named collection of `Array1` buffers
fn encode_buffers<A: Float>(
    buffers: &HashMap<String, Array1<A>>,
    prefix: &str,
    target: &mut HashMap<String, Vec<u8>>,
) -> Result<()> {
    for (name, buffer) in buffers.iter() {
        let key = format!("{}{}", prefix, name);
        let mut values = Vec::with_capacity(buffer.len());
        for value in buffer.iter() {
            values.push(to_state_value(&key, *value)?);
        }
        target.insert(key, encode_f64_slice(&values));
    }
    Ok(())
}

/// SGD optimizer with unified API
#[derive(Debug)]
pub struct UnifiedSGD<A: Float> {
    config: OptimizerConfig<A>,
    momentum_buffers: HashMap<String, Array1<A>>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> UnifiedSGD<A> {
    /// Create a new SGD optimizer
    pub fn new(config: OptimizerConfig<A>) -> Self {
        Self {
            config,
            momentum_buffers: HashMap::new(),
        }
    }

    /// Create SGD with momentum
    pub fn with_momentum(mut config: OptimizerConfig<A>, momentum: A) -> Self {
        config.params.insert("momentum".to_string(), momentum);
        Self::new(config)
    }
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> UnifiedOptimizer<A> for UnifiedSGD<A> {
    fn config(&self) -> &OptimizerConfig<A> {
        &self.config
    }

    fn step_param<D: Dimension>(&mut self, param: &mut Parameter<A, D>) -> Result<()> {
        if !param.requires_grad {
            return Ok(());
        }

        // Check gradient exists first
        if param.grad.is_none() {
            return Err(OptimError::InvalidConfig(
                "Parameter has no gradient".to_string(),
            ));
        }

        // Apply gradient clipping if configured
        if let Some(max_norm) = self.config.grad_clip {
            param.clip_grad(max_norm)?;
        }

        // Apply weight decay
        if self.config.weight_decay > A::zero() {
            param
                .data
                .mapv_inplace(|x| x * (A::one() - self.config.weight_decay * self.config.lr));
        }

        // Get gradient safely
        let grad = param.grad.as_ref().expect("unwrap failed");

        // Get momentum factor
        let momentum = self
            .config
            .params
            .get("momentum")
            .copied()
            .unwrap_or(A::zero());

        if momentum > A::zero() {
            // SGD with momentum
            if let Some(momentum_buffer) = self.momentum_buffers.get_mut(&param.name) {
                // Update momentum buffer
                for (m, g) in momentum_buffer.iter_mut().zip(grad.iter()) {
                    *m = momentum * (*m) + *g;
                }
                // Update parameters
                for (p, m) in param.data.iter_mut().zip(momentum_buffer.iter()) {
                    *p = *p - self.config.lr * (*m);
                }
            } else {
                // Initialize momentum buffer
                let mut momentum_buffer = Array1::zeros(grad.len());
                for (m, g) in momentum_buffer.iter_mut().zip(grad.iter()) {
                    *m = *g;
                }
                // Update parameters
                for (p, m) in param.data.iter_mut().zip(momentum_buffer.iter()) {
                    *p = *p - self.config.lr * (*m);
                }
                self.momentum_buffers
                    .insert(param.name.clone(), momentum_buffer);
            }
        } else {
            // Standard SGD
            for (p, g) in param.data.iter_mut().zip(grad.iter()) {
                *p = *p - self.config.lr * (*g);
            }
        }

        Ok(())
    }

    fn set_lr(&mut self, lr: A) {
        self.config.lr = lr;
    }

    fn get_lr(&self) -> A {
        self.config.lr
    }

    fn state_dict(&self) -> Result<HashMap<String, Vec<u8>>> {
        let mut state = HashMap::new();
        encode_config(&self.config, &mut state)?;
        encode_buffers(&self.momentum_buffers, "sgd.momentum_buffer.", &mut state)?;
        Ok(state)
    }

    fn load_state_dict(&mut self, statedict: HashMap<String, Vec<u8>>) -> Result<()> {
        decode_config(&statedict, &mut self.config)?;
        self.momentum_buffers =
            decode_buffers(&statedict, "sgd.momentum_buffer.", &self.momentum_buffers)?;
        Ok(())
    }
}

/// Adam optimizer with unified API
#[derive(Debug)]
pub struct UnifiedAdam<A: Float> {
    config: OptimizerConfig<A>,
    /// Per-parameter update counters driving bias correction
    ///
    /// Adam's bias correction depends on how many updates *that particular tensor*
    /// has received. A single shared counter would advance once per parameter in the
    /// model, so a 100-tensor model would reach t = 100 after a single optimizer
    /// step and its bias correction would be wrong for every tensor.
    step_counts: HashMap<String, usize>,
    exp_avg: HashMap<String, Array1<A>>,
    exp_avg_sq: HashMap<String, Array1<A>>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> UnifiedAdam<A> {
    /// Create a new Adam optimizer
    pub fn new(config: OptimizerConfig<A>) -> Self {
        let mut params = config.params.clone();
        params
            .entry("beta1".to_string())
            .or_insert_with(|| A::from(0.9).expect("unwrap failed"));
        params
            .entry("beta2".to_string())
            .or_insert_with(|| A::from(0.999).expect("unwrap failed"));
        params
            .entry("eps".to_string())
            .or_insert_with(|| A::from(1e-8).expect("unwrap failed"));

        Self {
            config: OptimizerConfig { params, ..config },
            step_counts: HashMap::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }

    /// Number of updates applied to the parameter called `name`
    pub fn step_count(&self, name: &str) -> usize {
        self.step_counts.get(name).copied().unwrap_or(0)
    }

    /// Create Adam with custom betas
    pub fn with_betas(mut config: OptimizerConfig<A>, beta1: A, beta2: A) -> Self {
        config.params.insert("beta1".to_string(), beta1);
        config.params.insert("beta2".to_string(), beta2);
        Self::new(config)
    }
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> UnifiedOptimizer<A> for UnifiedAdam<A> {
    fn config(&self) -> &OptimizerConfig<A> {
        &self.config
    }

    fn step_param<D: Dimension>(&mut self, param: &mut Parameter<A, D>) -> Result<()> {
        if !param.requires_grad {
            return Ok(());
        }

        // Check gradient exists first
        if param.grad.is_none() {
            return Err(OptimError::InvalidConfig(
                "Parameter has no gradient".to_string(),
            ));
        }

        // Apply gradient clipping if configured
        if let Some(max_norm) = self.config.grad_clip {
            param.clip_grad(max_norm)?;
        }

        // Advance this parameter's own clock, not a counter shared by every tensor.
        let step_count = {
            let counter = self.step_counts.entry(param.name.clone()).or_insert(0);
            *counter = counter.saturating_add(1);
            *counter
        };

        let beta1 = *self.config.params.get("beta1").ok_or_else(|| {
            OptimError::InvalidConfig("Adam configuration is missing 'beta1'".to_string())
        })?;
        let beta2 = *self.config.params.get("beta2").ok_or_else(|| {
            OptimError::InvalidConfig("Adam configuration is missing 'beta2'".to_string())
        })?;
        let eps = *self.config.params.get("eps").ok_or_else(|| {
            OptimError::InvalidConfig("Adam configuration is missing 'eps'".to_string())
        })?;

        // Get gradient safely
        let grad = param
            .grad
            .as_ref()
            .ok_or_else(|| OptimError::InvalidConfig("Parameter has no gradient".to_string()))?;

        // Initialize or get existing moment estimates
        let exp_avg = self
            .exp_avg
            .entry(param.name.clone())
            .or_insert_with(|| Array1::zeros(grad.len()));
        let exp_avg_sq = self
            .exp_avg_sq
            .entry(param.name.clone())
            .or_insert_with(|| Array1::zeros(grad.len()));

        // Update biased first and second moment estimates
        for ((exp_avg_val, exp_avg_sq_val), grad_val) in exp_avg
            .iter_mut()
            .zip(exp_avg_sq.iter_mut())
            .zip(grad.iter())
        {
            *exp_avg_val = beta1 * (*exp_avg_val) + (A::one() - beta1) * (*grad_val);
            *exp_avg_sq_val =
                beta2 * (*exp_avg_sq_val) + (A::one() - beta2) * (*grad_val) * (*grad_val);
        }

        // Bias correction driven by this parameter's own step count
        let exponent = i32::try_from(step_count).map_err(|_| {
            OptimError::InvalidConfig(
                "Timestep too large for bias correction calculation".to_string(),
            )
        })?;
        let bias_correction1 = A::one() - beta1.powi(exponent);
        let bias_correction2 = A::one() - beta2.powi(exponent);

        let step_size = self.config.lr * (bias_correction2.sqrt() / bias_correction1);

        // Update parameters
        for ((p, exp_avg_val), exp_avg_sq_val) in param
            .data
            .iter_mut()
            .zip(exp_avg.iter())
            .zip(exp_avg_sq.iter())
        {
            let denom = exp_avg_sq_val.sqrt() + eps;
            *p = *p - step_size * (*exp_avg_val) / denom;
        }

        // Apply weight decay after the main update
        if self.config.weight_decay > A::zero() {
            param
                .data
                .mapv_inplace(|x| x * (A::one() - self.config.weight_decay * self.config.lr));
        }

        Ok(())
    }

    fn set_lr(&mut self, lr: A) {
        self.config.lr = lr;
    }

    fn get_lr(&self) -> A {
        self.config.lr
    }

    fn state_dict(&self) -> Result<HashMap<String, Vec<u8>>> {
        let mut state = HashMap::new();
        encode_config(&self.config, &mut state)?;
        encode_buffers(&self.exp_avg, "adam.exp_avg.", &mut state)?;
        encode_buffers(&self.exp_avg_sq, "adam.exp_avg_sq.", &mut state)?;
        for (name, count) in self.step_counts.iter() {
            state.insert(
                format!("adam.step_count.{}", name),
                encode_f64_slice(&[*count as f64]),
            );
        }
        Ok(state)
    }

    fn load_state_dict(&mut self, statedict: HashMap<String, Vec<u8>>) -> Result<()> {
        decode_config(&statedict, &mut self.config)?;

        let exp_avg = decode_buffers(&statedict, "adam.exp_avg.", &self.exp_avg)?;
        let exp_avg_sq = decode_buffers(&statedict, "adam.exp_avg_sq.", &self.exp_avg_sq)?;

        // The two moment buffers describe the same tensors and must agree.
        for (name, buffer) in exp_avg.iter() {
            match exp_avg_sq.get(name) {
                Some(other) if other.len() == buffer.len() => {}
                Some(other) => {
                    return Err(OptimError::DimensionMismatch(format!(
                        "state dict moments for '{}' disagree: {} vs {} elements",
                        name,
                        buffer.len(),
                        other.len()
                    )))
                }
                None => {
                    return Err(OptimError::InvalidConfig(format!(
                        "state dict has 'adam.exp_avg.{}' but no matching 'adam.exp_avg_sq' entry",
                        name
                    )))
                }
            }
        }

        let mut step_counts = HashMap::new();
        for (key, bytes) in statedict.iter() {
            if let Some(name) = key.strip_prefix("adam.step_count.") {
                let value = decode_scalar(key, bytes)?;
                if !value.is_finite() || value < 0.0 {
                    return Err(OptimError::InvalidConfig(format!(
                        "state dict entry '{}' holds an invalid step count {}",
                        key, value
                    )));
                }
                step_counts.insert(name.to_string(), value as usize);
            }
        }

        self.exp_avg = exp_avg;
        self.exp_avg_sq = exp_avg_sq;
        self.step_counts = step_counts;
        Ok(())
    }
}

/// Optimizer factory for creating optimizers with unified API
pub struct OptimizerFactory;

impl OptimizerFactory {
    /// Create SGD optimizer
    pub fn sgd<A: Float + ScalarOperand + Debug + Send + Sync>(
        config: OptimizerConfig<A>,
    ) -> UnifiedSGD<A> {
        UnifiedSGD::new(config)
    }

    /// Create Adam optimizer
    pub fn adam<A: Float + ScalarOperand + Debug + Send + Sync>(
        config: OptimizerConfig<A>,
    ) -> UnifiedAdam<A> {
        UnifiedAdam::new(config)
    }

    /// Create SGD with momentum
    pub fn sgd_momentum<A: Float + ScalarOperand + Debug + Send + Sync>(
        config: OptimizerConfig<A>,
        momentum: A,
    ) -> UnifiedSGD<A> {
        UnifiedSGD::with_momentum(config, momentum)
    }

    /// Create Adam with custom parameters
    pub fn adam_custom<A: Float + ScalarOperand + Debug + Send + Sync>(
        config: OptimizerConfig<A>,
        beta1: A,
        beta2: A,
    ) -> UnifiedAdam<A> {
        UnifiedAdam::with_betas(config, beta1, beta2)
    }
}

/// Training loop helper with unified API
pub struct TrainingLoop<A: Float, O: UnifiedOptimizer<A>> {
    optimizer: O,
    scheduler: Option<Box<dyn LearningRateScheduler<A>>>,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float + ScalarOperand + Debug, O: UnifiedOptimizer<A> + Send + Sync> TrainingLoop<A, O> {
    /// Create a new training loop
    pub fn new(optimizer: O) -> Self {
        Self {
            optimizer,
            scheduler: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a learning rate scheduler
    pub fn with_scheduler(mut self, scheduler: Box<dyn LearningRateScheduler<A>>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Perform one training step
    pub fn step<D: Dimension>(&mut self, params: &mut [Parameter<A, D>]) -> Result<()> {
        // Update parameters
        self.optimizer.step_params(params)?;

        // Update learning rate if scheduler is present
        if let Some(ref mut scheduler) = self.scheduler {
            let new_lr = scheduler.step();
            self.optimizer.set_lr(new_lr);
        }

        Ok(())
    }

    /// Zero gradients
    pub fn zero_grad<D: Dimension>(&self, params: &mut [Parameter<A, D>]) {
        for param in params.iter_mut() {
            param.grad = None;
        }
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> A {
        self.optimizer.get_lr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_unified_sgd() {
        let config = OptimizerConfig::new(0.1f64);
        let mut optimizer = UnifiedSGD::new(config);

        let mut param = Parameter::new(Array1::from_vec(vec![1.0, 2.0, 3.0]), "test_param");
        param.set_grad(Array1::from_vec(vec![0.1, 0.2, 0.3]));

        optimizer.step_param(&mut param).expect("unwrap failed");

        // Check that parameters were updated correctly
        assert!((param.data[0] - 0.99).abs() < 1e-10);
        assert!((param.data[1] - 1.98).abs() < 1e-10);
        assert!((param.data[2] - 2.97).abs() < 1e-10);
    }

    #[test]
    fn test_unified_adam() {
        let config = OptimizerConfig::new(0.001f64);
        let mut optimizer = UnifiedAdam::new(config);

        let mut param = Parameter::new(Array1::from_vec(vec![1.0, 2.0, 3.0]), "test_param");
        param.set_grad(Array1::from_vec(vec![0.1, 0.2, 0.3]));

        optimizer.step_param(&mut param).expect("unwrap failed");

        // Parameters should have been updated (exact values depend on Adam's internal state)
        assert!(param.data[0] < 1.0);
        assert!(param.data[1] < 2.0);
        assert!(param.data[2] < 3.0);
    }

    #[test]
    fn test_optimizer_factory() {
        let config = OptimizerConfig::new(0.01f64).weight_decay(0.0001);
        let _sgd = OptimizerFactory::sgd(config.clone());
        let _adam = OptimizerFactory::adam(config);
    }

    #[test]
    fn test_parameter_operations() {
        let mut param = Parameter::new(Array1::from_vec(vec![1.0, 2.0, 3.0]), "test");

        // Test gradient setting
        param.set_grad(Array1::from_vec(vec![0.1, 0.2, 0.3]));
        assert!(param.grad().is_some());

        // Test gradient clipping
        param.clip_grad(0.1).expect("unwrap failed");
        let grad = param.grad().expect("unwrap failed");
        let norm: f64 = grad.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 0.1).abs() < 1e-10);

        // Test zero grad
        param.zero_grad();
        assert!(param.grad().is_none());
    }

    /// Regression test for the shared Adam step counter.
    ///
    /// `step_count` used to be a single counter incremented once per *parameter*, so
    /// the second tensor updated in a training step was bias-corrected as if it were
    /// on its second update. Every tensor must keep its own clock.
    #[test]
    fn test_unified_adam_step_count_is_per_parameter() {
        let config = OptimizerConfig::new(0.1f64);
        let mut optimizer = UnifiedAdam::new(config);

        let mut first = Parameter::new(Array1::from_vec(vec![0.0f64]), "layer1.weight");
        first.set_grad(Array1::from_vec(vec![1.0f64]));
        let mut second = Parameter::new(Array1::from_vec(vec![0.0f64]), "layer2.weight");
        second.set_grad(Array1::from_vec(vec![1.0f64]));

        optimizer.step_param(&mut first).expect("first step failed");
        optimizer
            .step_param(&mut second)
            .expect("second step failed");

        assert_eq!(optimizer.step_count("layer1.weight"), 1);
        assert_eq!(optimizer.step_count("layer2.weight"), 1);

        // At t = 1 with a unit gradient the Adam step is -lr up to the epsilon term
        // (denominator sqrt(v_hat) + eps), i.e. a relative error of about 3e-7.
        assert!((first.data[0] + 0.1).abs() < 1e-6, "got {}", first.data[0]);
        assert!(
            (second.data[0] + 0.1).abs() < 1e-6,
            "second tensor used the wrong timestep: {}",
            second.data[0]
        );
        assert!((first.data[0] - second.data[0]).abs() < 1e-12);
    }

    /// A round trip through the state dictionary must reproduce the exact trajectory.
    #[test]
    fn test_unified_adam_state_dict_round_trip() {
        let config = OptimizerConfig::new(0.05f64).weight_decay(0.01);
        let mut original = UnifiedAdam::new(config.clone());

        let mut param = Parameter::new(Array1::from_vec(vec![1.0f64, 2.0, 3.0]), "w");
        for i in 0..5 {
            let scale = 1.0 + i as f64;
            param.set_grad(Array1::from_vec(vec![0.1 * scale, -0.2, 0.3]));
            original.step_param(&mut param).expect("step failed");
        }

        let state = original.state_dict().expect("state_dict failed");
        assert!(!state.is_empty(), "state dict must not be empty");
        assert!(state.contains_key("adam.exp_avg.w"));
        assert!(state.contains_key("adam.exp_avg_sq.w"));
        assert!(state.contains_key("adam.step_count.w"));

        let mut restored = UnifiedAdam::new(OptimizerConfig::new(999.0f64));
        restored
            .load_state_dict(state)
            .expect("load_state_dict failed");

        assert_eq!(restored.step_count("w"), 5);
        assert!((restored.get_lr() - 0.05).abs() < 1e-12);

        // Continue both optimizers from the same parameters and compare.
        let mut a = param.clone();
        let mut b = param.clone();
        a.set_grad(Array1::from_vec(vec![0.4f64, -0.2, 0.3]));
        b.set_grad(Array1::from_vec(vec![0.4f64, -0.2, 0.3]));
        original.step_param(&mut a).expect("continue original");
        restored.step_param(&mut b).expect("continue restored");

        for i in 0..3 {
            assert!(
                (a.data[i] - b.data[i]).abs() < 1e-12,
                "restored optimizer diverged at {}: {} vs {}",
                i,
                a.data[i],
                b.data[i]
            );
        }
    }

    /// Loading a checkpoint whose buffers do not match must be rejected, not ignored.
    #[test]
    fn test_unified_adam_load_state_dict_validates_shapes() {
        let mut optimizer = UnifiedAdam::new(OptimizerConfig::new(0.1f64));
        let mut param = Parameter::new(Array1::from_vec(vec![1.0f64, 2.0, 3.0]), "w");
        param.set_grad(Array1::from_vec(vec![0.1f64, 0.2, 0.3]));
        optimizer.step_param(&mut param).expect("step failed");

        let mut state = optimizer.state_dict().expect("state_dict failed");

        // Shrink one moment buffer: the optimizer already holds three elements.
        let mut truncated = state
            .get("adam.exp_avg.w")
            .cloned()
            .expect("exp_avg entry must exist");
        truncated.truncate(8);
        state.insert("adam.exp_avg.w".to_string(), truncated);

        assert!(optimizer.load_state_dict(state.clone()).is_err());

        // A payload that is not a whole number of f64 values is rejected too.
        state.insert("adam.exp_avg.w".to_string(), vec![0u8; 7]);
        assert!(optimizer.load_state_dict(state).is_err());
    }

    /// SGD momentum buffers survive a state-dict round trip.
    #[test]
    fn test_unified_sgd_state_dict_round_trip() {
        let config = OptimizerConfig::new(0.1f64);
        let mut original = UnifiedSGD::with_momentum(config, 0.9);

        let mut param = Parameter::new(Array1::from_vec(vec![1.0f64, 2.0]), "w");
        for _ in 0..3 {
            param.set_grad(Array1::from_vec(vec![0.1f64, 0.2]));
            original.step_param(&mut param).expect("step failed");
        }

        let state = original.state_dict().expect("state_dict failed");
        assert!(state.contains_key("sgd.momentum_buffer.w"));

        let mut restored = UnifiedSGD::new(OptimizerConfig::new(999.0f64));
        restored
            .load_state_dict(state)
            .expect("load_state_dict failed");
        assert!((restored.get_lr() - 0.1).abs() < 1e-12);

        let mut a = param.clone();
        let mut b = param.clone();
        a.set_grad(Array1::from_vec(vec![0.1f64, 0.2]));
        b.set_grad(Array1::from_vec(vec![0.1f64, 0.2]));
        original.step_param(&mut a).expect("continue original");
        restored.step_param(&mut b).expect("continue restored");

        assert!((a.data[0] - b.data[0]).abs() < 1e-12);
        assert!((a.data[1] - b.data[1]).abs() < 1e-12);
    }

    /// A state dict without a recognised version header must be rejected.
    #[test]
    fn test_state_dict_version_is_checked() {
        let mut optimizer = UnifiedSGD::new(OptimizerConfig::new(0.1f64));
        let mut state = HashMap::new();
        state.insert("config.lr".to_string(), 0.5f64.to_le_bytes().to_vec());
        assert!(optimizer.load_state_dict(state.clone()).is_err());

        state.insert("format.version".to_string(), 7.0f64.to_le_bytes().to_vec());
        assert!(optimizer.load_state_dict(state).is_err());
    }
}
