//! Neural Turing Machine (NTM) Memory-Augmented Learned Optimizer
//!
//! This module implements a learned optimizer whose update rule is produced by a
//! Neural Turing Machine: a feedforward controller coupled to an external memory
//! matrix through a differentiable read/write head that uses the full content- and
//! location-based addressing mechanism of Graves, Wayne & Danihelka (2014),
//! *Neural Turing Machines* (arXiv:1410.5401), applied to the "learning to
//! optimize" setting (Andrychowicz et al., 2016).
//!
//! # Overview
//!
//! The optimizer treats the flat parameter vector as a temporal stream. Each
//! optimization [`AdvancedOptimizer::step`] partitions the parameters into
//! `num_slots` contiguous chunks (clamped to the parameter count). Every chunk is
//! processed in turn as one controller time-step, and the external memory plus the
//! addressing weights and read vector persist across both chunks and steps, giving
//! the optimizer a genuine recurrent memory.
//!
//! For each chunk the optimizer:
//!
//! 1. **Features.** Computes aggregate gradient statistics over the chunk (mean,
//!    abs-mean, variance, bias-corrected momentum, bias-corrected RMS and sign
//!    agreement) and concatenates them with the previous read vector to form the
//!    controller input.
//! 2. **Controller.** Maps that input through a real feedforward layer
//!    (`tanh(W_in x + b_in)`) and emits every head parameter through dedicated
//!    projection heads: the content key `k`, key strength `β = softplus(·)`,
//!    interpolation gate `g = σ(·)`, shift distribution `s = softmax(·)`,
//!    sharpening `γ = 1 + softplus(·)`, erase vector `e = σ(·)` and add vector
//!    `a = tanh(·)`.
//! 3. **Addressing.** Runs the complete pipeline — content addressing
//!    (`w_c = softmax(β · cosine(k, M))`), interpolation
//!    (`w_g = g·w_c + (1-g)·w_prev`), circular convolutional shift
//!    (`w_s[i] = Σ_d w_g[(i-d) mod N] · s_d`) and sharpening
//!    (`w[i] = w_s[i]^γ / Σ_j w_s[j]^γ`).
//! 4. **Read / write.** Reads `r = Σ_i w[i]·M[i]` from memory, then writes with the
//!    erase/add rule `M[i] ← M[i]∘(1 - w[i]·e) + w[i]·a`.
//! 5. **Update.** Combines the controller hidden state and the read vector into a
//!    strictly-positive per-chunk step scale in `(0, 1)`, and applies
//!    `params[chunk] -= base_lr · scale · m̂[chunk]`, where `m̂` is the
//!    bias-corrected EMA-smoothed gradient (the descent direction).
//!
//! The controller weights and the initial memory are initialised deterministically
//! from a seed via [`scirs2_core::random`] and held fixed during
//! [`AdvancedOptimizer::step`] (this is the genuine inference optimizer; the memory,
//! addressing weights, read vector and gradient EMAs are persistent state that
//! evolves across steps). The architecture is laid out so that meta-training of the
//! controller weights could be added later without changing the forward path.

use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::{Random, Rng};
use std::fmt::Debug;

use crate::domain_optimizers::{clip_grad_norm, l2_norm, AdvancedOptimizer, OptimizerStateInfo};
use crate::error::{OptimError, Result};

/// Number of scalar gradient features extracted per chunk, every step.
///
/// In order: mean gradient, abs-mean gradient, gradient variance, momentum
/// (bias-corrected gradient EMA), RMS (sqrt of bias-corrected squared-gradient
/// EMA) and sign agreement.
const CHUNK_FEATURE_DIM: usize = 6;

/// Configuration for [`NtmOptimizer`].
///
/// Numeric hyper-parameters are stored as `f64` and converted to the optimizer's
/// scalar type `T` when the optimizer is constructed.
#[derive(Debug, Clone)]
pub struct NtmOptimizerConfig {
    /// Requested number of parameter chunks (clamped to the parameter count).
    pub num_slots: usize,
    /// Number of memory locations (rows of the memory matrix `M`).
    pub num_locations: usize,
    /// Width of each memory location (columns of `M`); also the read/key width.
    pub mem_width: usize,
    /// Controller hidden-layer dimension.
    pub hidden_dim: usize,
    /// Maximum convolutional shift magnitude; allowed shifts are
    /// `-shift_range ..= shift_range`, giving `2·shift_range + 1` entries.
    pub shift_range: usize,
    /// Half-width of the uniform interval used to initialise the memory matrix.
    pub mem_init_scale: f64,
    /// Base learning rate applied before the learned per-chunk step scale.
    pub base_lr: f64,
    /// EMA decay for the gradient (momentum), often called `beta1`.
    pub momentum_decay: f64,
    /// EMA decay for the squared gradient (RMS), often called `beta2`.
    pub rms_decay: f64,
    /// Maximum gradient L2 norm; larger gradients are rescaled before use.
    pub max_grad_norm: f64,
    /// Seed for deterministic weight and memory initialisation.
    pub seed: u64,
}

impl Default for NtmOptimizerConfig {
    fn default() -> Self {
        Self {
            num_slots: 8,
            num_locations: 16,
            mem_width: 8,
            hidden_dim: 16,
            shift_range: 1,
            mem_init_scale: 0.1,
            base_lr: 0.01,
            momentum_decay: 0.9,
            rms_decay: 0.999,
            max_grad_norm: 10.0,
            seed: 0x47A5_E500,
        }
    }
}

/// Convert an `f64` into the optimizer scalar, falling back to zero on overflow.
fn cast<T: Float>(value: f64) -> T {
    T::from(value).unwrap_or_else(T::zero)
}

/// Numerically stable logistic (sigmoid) activation.
fn sigmoid<T: Float>(x: T) -> T {
    if x >= T::zero() {
        T::one() / (T::one() + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (T::one() + ex)
    }
}

/// Numerically stable softplus, `ln(1 + e^x) ≥ 0`.
fn softplus<T: Float>(x: T) -> T {
    x.max(T::zero()) + (T::one() + (-x.abs()).exp()).ln()
}

/// Numerically stable softmax over a 1-D logit vector.
fn softmax<T: Float>(logits: &Array1<T>) -> Array1<T> {
    let max = logits
        .iter()
        .fold(T::neg_infinity(), |m, &x| if x > m { x } else { m });
    let max = if max.is_finite() { max } else { T::zero() };
    let exps = logits.mapv(|x| (x - max).exp());
    let sum = exps.iter().fold(T::zero(), |a, &x| a + x);
    let denom = if sum > T::zero() { sum } else { T::one() };
    exps.mapv(|x| x / denom)
}

/// Glorot-uniform initialisation of a `rows x cols` weight matrix.
fn init_matrix<T, R>(rng: &mut Random<R>, rows: usize, cols: usize) -> Array2<T>
where
    T: Float,
    R: Rng,
{
    let limit = (6.0 / (rows as f64 + cols as f64)).sqrt();
    Array2::from_shape_fn((rows, cols), |_| {
        let value: f64 = rng.random_range(-limit..limit);
        cast(value)
    })
}

/// Glorot-uniform initialisation of a length-`len` vector.
fn init_vector<T, R>(rng: &mut Random<R>, len: usize) -> Array1<T>
where
    T: Float,
    R: Rng,
{
    let limit = (6.0 / (len as f64 + 1.0)).sqrt();
    Array1::from_shape_fn(len, |_| {
        let value: f64 = rng.random_range(-limit..limit);
        cast(value)
    })
}

/// Uniform initialisation of a `rows x cols` matrix in `(-scale, scale)`.
fn init_matrix_uniform<T, R>(rng: &mut Random<R>, rows: usize, cols: usize, scale: f64) -> Array2<T>
where
    T: Float,
    R: Rng,
{
    let limit = scale.abs().max(f64::MIN_POSITIVE);
    Array2::from_shape_fn((rows, cols), |_| {
        let value: f64 = rng.random_range(-limit..limit);
        cast(value)
    })
}

/// The full set of head parameters emitted by the controller for one time-step.
#[derive(Debug, Clone)]
struct HeadParams<T: Float + Debug + Send + Sync + 'static> {
    /// Content key (length `mem_width`).
    key: Array1<T>,
    /// Key strength `β ≥ 0`.
    beta: T,
    /// Interpolation gate `g ∈ [0, 1]`.
    gate: T,
    /// Shift distribution over the allowed shifts (length `2·shift_range + 1`).
    shift: Array1<T>,
    /// Sharpening exponent `γ ≥ 1`.
    gamma: T,
    /// Erase vector `e ∈ [0, 1]^width`.
    erase: Array1<T>,
    /// Add vector `a` (length `mem_width`).
    add: Array1<T>,
}

/// A genuine feedforward NTM controller with one hidden layer.
///
/// The controller maps a feature/read input through `tanh(W_in x + b_in)` and then
/// emits every addressing-head parameter and an output-head step scale through
/// dedicated linear projections with the appropriate activations. All weights are
/// fixed during inference; they are exposed only through forward methods so that a
/// meta-training loop could optimise them without altering this forward path.
#[derive(Debug, Clone)]
struct NtmController<T: Float + Debug + Send + Sync + 'static> {
    w_in: Array2<T>,
    b_in: Array1<T>,
    w_key: Array2<T>,
    b_key: Array1<T>,
    w_beta: Array1<T>,
    b_beta: T,
    w_gate: Array1<T>,
    b_gate: T,
    w_shift: Array2<T>,
    b_shift: Array1<T>,
    w_gamma: Array1<T>,
    b_gamma: T,
    w_erase: Array2<T>,
    b_erase: Array1<T>,
    w_add: Array2<T>,
    b_add: Array1<T>,
    w_out: Array1<T>,
    v_out: Array1<T>,
    b_out: T,
}

impl<T: Float + Debug + Send + Sync + 'static> NtmController<T> {
    /// Deterministically initialise every controller weight.
    fn new<R: Rng>(
        rng: &mut Random<R>,
        input_dim: usize,
        hidden: usize,
        mem_width: usize,
        num_shifts: usize,
    ) -> Self {
        Self {
            w_in: init_matrix(rng, hidden, input_dim),
            b_in: Array1::zeros(hidden),
            w_key: init_matrix(rng, mem_width, hidden),
            b_key: Array1::zeros(mem_width),
            w_beta: init_vector(rng, hidden),
            b_beta: T::zero(),
            w_gate: init_vector(rng, hidden),
            b_gate: T::zero(),
            w_shift: init_matrix(rng, num_shifts, hidden),
            b_shift: Array1::zeros(num_shifts),
            w_gamma: init_vector(rng, hidden),
            b_gamma: T::zero(),
            w_erase: init_matrix(rng, mem_width, hidden),
            b_erase: Array1::zeros(mem_width),
            w_add: init_matrix(rng, mem_width, hidden),
            b_add: Array1::zeros(mem_width),
            w_out: init_vector(rng, hidden),
            v_out: init_vector(rng, mem_width),
            b_out: T::zero(),
        }
    }

    /// Compute the hidden state `tanh(W_in x + b_in)`.
    fn hidden_state(&self, input: &Array1<T>) -> Array1<T> {
        (self.w_in.dot(input) + &self.b_in).mapv(|x| x.tanh())
    }

    /// Emit every addressing-head parameter from the hidden state.
    fn heads(&self, hidden: &Array1<T>) -> HeadParams<T> {
        let key = self.w_key.dot(hidden) + &self.b_key;
        let beta = softplus(self.w_beta.dot(hidden) + self.b_beta);
        let gate = sigmoid(self.w_gate.dot(hidden) + self.b_gate);
        let shift = softmax(&(self.w_shift.dot(hidden) + &self.b_shift));
        let gamma = T::one() + softplus(self.w_gamma.dot(hidden) + self.b_gamma);
        let erase = (self.w_erase.dot(hidden) + &self.b_erase).mapv(sigmoid);
        let add = (self.w_add.dot(hidden) + &self.b_add).mapv(|x| x.tanh());
        HeadParams {
            key,
            beta,
            gate,
            shift,
            gamma,
            erase,
            add,
        }
    }

    /// Map the hidden state and the memory read into a step scale in `(0, 1)`.
    fn output_scale(&self, hidden: &Array1<T>, read: &Array1<T>) -> T {
        sigmoid(self.w_out.dot(hidden) + self.v_out.dot(read) + self.b_out)
    }
}

/// Content-based addressing: `w_c = softmax(β · cosine_similarity(k, M[i]))`.
fn content_addressing<T>(memory: &Array2<T>, key: &Array1<T>, beta: T) -> Array1<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    let eps = cast::<T>(1e-8);
    let key_norm = key.iter().fold(T::zero(), |acc, &x| acc + x * x).sqrt();
    let mut sims = Array1::<T>::zeros(memory.nrows());
    for (slot, row) in sims.iter_mut().zip(memory.outer_iter()) {
        let mut dot = T::zero();
        let mut row_sq = T::zero();
        for (&kv, &rv) in key.iter().zip(row.iter()) {
            dot = dot + kv * rv;
            row_sq = row_sq + rv * rv;
        }
        let row_norm = row_sq.sqrt();
        let cosine = dot / (key_norm * row_norm + eps);
        *slot = beta * cosine;
    }
    softmax(&sims)
}

/// Interpolation: `w_g = g · w_c + (1 - g) · w_prev`.
fn interpolate<T>(content: &Array1<T>, prev: &Array1<T>, gate: T) -> Array1<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    let one_minus = T::one() - gate;
    content.mapv(|x| x * gate) + &prev.mapv(|x| x * one_minus)
}

/// Circular convolutional shift: `w_s[i] = Σ_d w_g[(i - d) mod N] · s_d`.
fn circular_shift<T>(gated: &Array1<T>, shift: &Array1<T>, offsets: &[isize]) -> Array1<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    let n = gated.len();
    let n_isize = n as isize;
    let mut shifted = Array1::<T>::zeros(n);
    for (i, slot) in shifted.iter_mut().enumerate() {
        let mut acc = T::zero();
        for (idx, &offset) in offsets.iter().enumerate() {
            let src = (((i as isize - offset) % n_isize) + n_isize) % n_isize;
            acc = acc + gated[src as usize] * shift[idx];
        }
        *slot = acc;
    }
    shifted
}

/// Sharpening: `w[i] = w_s[i]^γ / Σ_j w_s[j]^γ`.
fn sharpen<T>(shifted: &Array1<T>, gamma: T) -> Array1<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    let powered = shifted.mapv(|x| {
        if x > T::zero() {
            x.powf(gamma)
        } else {
            T::zero()
        }
    });
    let sum = powered.iter().fold(T::zero(), |a, &x| a + x);
    let denom = if sum > T::zero() { sum } else { T::one() };
    powered.mapv(|x| x / denom)
}

/// Run the full addressing pipeline, returning the final (sharpened) weights.
fn address<T>(
    memory: &Array2<T>,
    heads: &HeadParams<T>,
    prev_weights: &Array1<T>,
    offsets: &[isize],
) -> Array1<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    let content = content_addressing(memory, &heads.key, heads.beta);
    let gated = interpolate(&content, prev_weights, heads.gate);
    let shifted = circular_shift(&gated, &heads.shift, offsets);
    sharpen(&shifted, heads.gamma)
}

/// Read from memory: `r = Σ_i w[i] · M[i]`.
fn read_memory<T>(memory: &Array2<T>, weights: &Array1<T>) -> Array1<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    weights
        .iter()
        .zip(memory.outer_iter())
        .fold(Array1::<T>::zeros(memory.ncols()), |acc, (&weight, row)| {
            acc + &row.mapv(|val| val * weight)
        })
}

/// Write to memory: `M[i] ← M[i] ∘ (1 - w[i]·e) + w[i]·a`.
fn write_memory<T>(memory: &mut Array2<T>, weights: &Array1<T>, erase: &Array1<T>, add: &Array1<T>)
where
    T: Float + Debug + Send + Sync + 'static,
{
    for (i, mut row) in memory.outer_iter_mut().enumerate() {
        let weight = weights[i];
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = *slot * (T::one() - weight * erase[j]) + weight * add[j];
        }
    }
}

/// Compute the aggregate gradient feature vector for one chunk.
fn chunk_features<T>(
    grad: &Array1<T>,
    m_hat: &Array1<T>,
    v_hat: &Array1<T>,
    start: usize,
    end: usize,
) -> Array1<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    let size = cast::<T>((end - start) as f64);
    let chunk = grad.slice(s![start..end]);

    let mean_grad = chunk.sum() / size;
    let abs_mean_grad = chunk.iter().fold(T::zero(), |a, &x| a + x.abs()) / size;
    let variance = chunk.iter().fold(T::zero(), |a, &x| {
        let d = x - mean_grad;
        a + d * d
    }) / size;

    let momentum = m_hat.slice(s![start..end]).sum() / size;
    let rms = v_hat
        .slice(s![start..end])
        .iter()
        .fold(T::zero(), |a, &x| a + x.max(T::zero()).sqrt())
        / size;

    let mean_sign = mean_grad.signum();
    let sign_agreement = chunk
        .iter()
        .fold(T::zero(), |a, &x| a + x.signum() * mean_sign)
        / size;

    Array1::from_vec(vec![
        mean_grad,
        abs_mean_grad,
        variance,
        momentum,
        rms,
        sign_agreement,
    ])
}

/// Concatenate two 1-D arrays into a new one.
fn concat<T>(head: &Array1<T>, tail: &Array1<T>) -> Array1<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    let mut values = Vec::with_capacity(head.len() + tail.len());
    values.extend(head.iter().copied());
    values.extend(tail.iter().copied());
    Array1::from_vec(values)
}

/// Partition `param_len` indices into `num_slots` contiguous, near-equal chunks.
fn build_chunks(param_len: usize, num_slots: usize) -> Vec<(usize, usize)> {
    let base = param_len / num_slots;
    let remainder = param_len % num_slots;
    let mut bounds = Vec::with_capacity(num_slots);
    let mut start = 0;
    for n in 0..num_slots {
        let size = base + usize::from(n < remainder);
        bounds.push((start, start + size));
        start += size;
    }
    bounds
}

/// Build the ordered list of allowed integer shift offsets `-range ..= range`.
fn build_shift_offsets(shift_range: usize) -> Vec<isize> {
    let range = shift_range as isize;
    (-range..=range).collect()
}

/// A memory-augmented learned optimizer driven by a Neural Turing Machine.
///
/// Implements [`AdvancedOptimizer`]. The controller weights and the initial memory
/// are fixed; the memory matrix, the previous addressing weights, the previous read
/// vector and the gradient EMAs are persistent state that evolves across steps.
#[derive(Debug, Clone)]
pub struct NtmOptimizer<T: Float + Debug + Send + Sync + 'static> {
    // --- configuration (fixed) ---
    requested_num_slots: usize,
    num_locations: usize,
    mem_width: usize,
    hidden_dim: usize,
    shift_range: usize,
    base_lr: T,
    momentum_decay: T,
    rms_decay: T,
    max_grad_norm: T,
    norm_ema_decay: T,
    seed: u64,
    shift_offsets: Vec<isize>,

    // --- learned controller (fixed during step) ---
    controller: NtmController<T>,

    // --- persistent memory + addressing state ---
    memory: Array2<T>,
    prev_weights: Array1<T>,
    prev_read: Array1<T>,
    last_weights: Array1<T>,

    // --- parameter layout (depends on the actual parameter length) ---
    param_len: usize,
    num_slots: usize,
    chunk_bounds: Vec<(usize, usize)>,

    // --- persistent optimization state ---
    grad_ema: Array1<T>,
    grad_sq_ema: Array1<T>,

    // --- bookkeeping ---
    step_count: usize,
    current_lr: T,
    grad_norm_ema: T,
}

impl<T: Float + Debug + Send + Sync + 'static> NtmOptimizer<T> {
    /// Construct a new optimizer from a configuration.
    ///
    /// The controller weights and the initial memory are initialised
    /// deterministically from `config.seed`. Parameter-dependent state is created
    /// lazily on the first [`AdvancedOptimizer::step`] once the parameter length is
    /// known.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidConfig`] if a hyper-parameter is out of range.
    pub fn new(config: NtmOptimizerConfig) -> Result<Self> {
        if config.num_slots == 0 {
            return Err(OptimError::InvalidConfig(
                "num_slots must be at least 1".to_string(),
            ));
        }
        if config.num_locations == 0 {
            return Err(OptimError::InvalidConfig(
                "num_locations must be at least 1".to_string(),
            ));
        }
        if config.mem_width == 0 {
            return Err(OptimError::InvalidConfig(
                "mem_width must be at least 1".to_string(),
            ));
        }
        if config.hidden_dim == 0 {
            return Err(OptimError::InvalidConfig(
                "hidden_dim must be at least 1".to_string(),
            ));
        }
        if config.shift_range == 0 {
            return Err(OptimError::InvalidConfig(
                "shift_range must be at least 1".to_string(),
            ));
        }
        if !(config.mem_init_scale.is_finite() && config.mem_init_scale > 0.0) {
            return Err(OptimError::InvalidConfig(
                "mem_init_scale must be a positive finite value".to_string(),
            ));
        }
        if !(config.base_lr.is_finite() && config.base_lr > 0.0) {
            return Err(OptimError::InvalidConfig(
                "base_lr must be a positive finite value".to_string(),
            ));
        }
        if !(config.max_grad_norm.is_finite() && config.max_grad_norm > 0.0) {
            return Err(OptimError::InvalidConfig(
                "max_grad_norm must be a positive finite value".to_string(),
            ));
        }
        for (name, value) in [
            ("momentum_decay", config.momentum_decay),
            ("rms_decay", config.rms_decay),
        ] {
            if !(0.0..1.0).contains(&value) {
                return Err(OptimError::InvalidConfig(format!(
                    "{name} must lie in [0, 1), got {value}"
                )));
            }
        }

        let num_shifts = 2 * config.shift_range + 1;
        let input_dim = CHUNK_FEATURE_DIM + config.mem_width;

        let mut rng = Random::seed(config.seed);
        let controller = NtmController::new(
            &mut rng,
            input_dim,
            config.hidden_dim,
            config.mem_width,
            num_shifts,
        );
        let memory = init_matrix_uniform(
            &mut rng,
            config.num_locations,
            config.mem_width,
            config.mem_init_scale,
        );

        let uniform = cast::<T>(1.0 / config.num_locations as f64);
        let prev_weights = Array1::from_elem(config.num_locations, uniform);

        Ok(Self {
            requested_num_slots: config.num_slots,
            num_locations: config.num_locations,
            mem_width: config.mem_width,
            hidden_dim: config.hidden_dim,
            shift_range: config.shift_range,
            base_lr: cast(config.base_lr),
            momentum_decay: cast(config.momentum_decay),
            rms_decay: cast(config.rms_decay),
            max_grad_norm: cast(config.max_grad_norm),
            norm_ema_decay: cast(0.99),
            seed: config.seed,
            shift_offsets: build_shift_offsets(config.shift_range),
            controller,
            memory,
            prev_weights: prev_weights.clone(),
            prev_read: Array1::zeros(config.mem_width),
            last_weights: prev_weights,
            param_len: 0,
            num_slots: 0,
            chunk_bounds: Vec::new(),
            grad_ema: Array1::zeros(0),
            grad_sq_ema: Array1::zeros(0),
            step_count: 0,
            current_lr: cast(config.base_lr),
            grad_norm_ema: T::zero(),
        })
    }

    /// The actual number of parameter chunks (after clamping to the parameter
    /// count). Returns `0` before the first step.
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// The number of memory locations (rows of `M`).
    pub fn num_locations(&self) -> usize {
        self.num_locations
    }

    /// The memory width (columns of `M`).
    pub fn mem_width(&self) -> usize {
        self.mem_width
    }

    /// The controller hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// The shift range (allowed shifts are `-range ..= range`).
    pub fn shift_range(&self) -> usize {
        self.shift_range
    }

    /// The seed used for deterministic initialisation.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// A read-only view of the external memory matrix `M`.
    pub fn memory(&self) -> &Array2<T> {
        &self.memory
    }

    /// The most recent (sharpened) addressing weight distribution.
    pub fn last_address_weights(&self) -> &Array1<T> {
        &self.last_weights
    }

    /// (Re)initialise parameter-dependent layout and EMAs for `param_len`.
    fn initialize_for(&mut self, param_len: usize) {
        let num_slots = self.requested_num_slots.min(param_len).max(1);
        self.param_len = param_len;
        self.num_slots = num_slots;
        self.chunk_bounds = build_chunks(param_len, num_slots);
        self.grad_ema = Array1::zeros(param_len);
        self.grad_sq_ema = Array1::zeros(param_len);
        self.step_count = 0;
        self.grad_norm_ema = T::zero();
    }
}

impl<T: Float + Debug + Send + Sync + 'static> AdvancedOptimizer<T> for NtmOptimizer<T> {
    fn step(&mut self, params: &Array1<T>, gradients: &Array1<T>) -> Result<Array1<T>> {
        if params.len() != gradients.len() {
            return Err(OptimError::InvalidConfig(format!(
                "Parameter length {} != gradient length {}",
                params.len(),
                gradients.len()
            )));
        }
        if params.is_empty() {
            return Err(OptimError::InsufficientData(
                "Empty parameter array".to_string(),
            ));
        }

        if self.param_len != params.len() {
            self.initialize_for(params.len());
        }

        // 1. Clip the gradient and track its EMA norm.
        let grad = clip_grad_norm(gradients, self.max_grad_norm);
        let grad_norm = l2_norm(&grad);
        self.grad_norm_ema =
            self.norm_ema_decay * self.grad_norm_ema + (T::one() - self.norm_ema_decay) * grad_norm;

        // 2. Update the per-element gradient and squared-gradient EMAs.
        let beta1 = self.momentum_decay;
        let beta2 = self.rms_decay;
        self.grad_ema = self.grad_ema.mapv(|m| m * beta1) + &grad.mapv(|g| g * (T::one() - beta1));
        self.grad_sq_ema =
            self.grad_sq_ema.mapv(|v| v * beta2) + &grad.mapv(|g| g * g * (T::one() - beta2));

        // 3. Bias-correct the EMAs (Adam-style), giving the effective direction.
        let step = self.step_count + 1;
        let bias1 = T::one() - beta1.powi(step as i32);
        let bias2 = T::one() - beta2.powi(step as i32);
        let m_hat = self.grad_ema.mapv(|m| m / bias1);
        let v_hat = self.grad_sq_ema.mapv(|v| v / bias2);

        // 4. Process every chunk as one controller time-step, threading the
        //    persistent memory, addressing weights and read vector through.
        let base_lr = self.base_lr;
        let mut scale_vector = Array1::<T>::zeros(self.param_len);
        let mut scale_sum = T::zero();

        for c in 0..self.num_slots {
            let (start, end) = self.chunk_bounds[c];

            let features = chunk_features(&grad, &m_hat, &v_hat, start, end);
            let input = concat(&features, &self.prev_read);

            let hidden = self.controller.hidden_state(&input);
            let heads = self.controller.heads(&hidden);

            let weights = address(
                &self.memory,
                &heads,
                &self.prev_weights,
                &self.shift_offsets,
            );
            let read = read_memory(&self.memory, &weights);
            let scale = self.controller.output_scale(&hidden, &read);
            write_memory(&mut self.memory, &weights, &heads.erase, &heads.add);

            self.prev_weights = weights.clone();
            self.last_weights = weights;
            self.prev_read = read;

            scale_vector.slice_mut(s![start..end]).fill(scale);
            scale_sum = scale_sum + scale;
        }

        // 5. Apply the per-chunk-scaled, EMA-smoothed update.
        let update = (&scale_vector * &m_hat).mapv(|u| u * base_lr);
        let new_params = params - &update;

        // 6. Effective learning rate = base_lr · mean chunk scale; advance counter.
        let mean_scale = scale_sum / cast::<T>(self.num_slots as f64);
        self.current_lr = base_lr * mean_scale;
        self.step_count += 1;

        Ok(new_params)
    }

    fn get_learning_rate(&self) -> T {
        self.current_lr
    }

    fn set_learning_rate(&mut self, lr: T) {
        self.base_lr = lr;
        self.current_lr = lr;
    }

    fn name(&self) -> &str {
        "NtmOptimizer"
    }

    fn get_state(&self) -> OptimizerStateInfo<T> {
        OptimizerStateInfo {
            step_count: self.step_count,
            current_lr: self.current_lr,
            grad_norm_ema: self.grad_norm_ema,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_optimizer(seed: u64) -> NtmOptimizer<f64> {
        NtmOptimizer::new(NtmOptimizerConfig {
            seed,
            ..Default::default()
        })
        .expect("default configuration must be valid")
    }

    fn l2(a: &Array1<f64>) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn is_distribution(weights: &Array1<f64>) -> bool {
        let non_negative = weights.iter().all(|&w| w >= -1e-12);
        let sum: f64 = weights.iter().sum();
        non_negative && (sum - 1.0).abs() < 1e-9
    }

    fn entropy(weights: &Array1<f64>) -> f64 {
        weights
            .iter()
            .filter(|&&w| w > 1e-15)
            .map(|&w| -w * w.ln())
            .sum()
    }

    #[test]
    fn test_ntm_preserves_shape() {
        for len in [1usize, 3, 7, 16, 40, 100] {
            let mut opt = make_optimizer(7);
            let params = Array1::from_shape_fn(len, |i| (i as f64) * 0.1 + 0.5);
            let grads = Array1::from_shape_fn(len, |i| (i as f64) * 0.01 + 0.2);
            let out = opt.step(&params, &grads).expect("step should succeed");
            assert_eq!(out.len(), params.len(), "output length must match input");
        }
    }

    #[test]
    fn test_ntm_deterministic_same_seed() {
        let params = Array1::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, 0.5, 0.25, 0.1]);
        let grads = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.05, 0.02, 0.3]);

        let mut a = make_optimizer(321);
        let mut b = make_optimizer(321);
        let out_a = a.step(&params, &grads).expect("a step");
        let out_b = b.step(&params, &grads).expect("b step");

        for (x, y) in out_a.iter().zip(out_b.iter()) {
            assert!(
                (x - y).abs() < 1e-12,
                "same seed must give identical output: {x} vs {y}"
            );
        }
    }

    #[test]
    fn test_ntm_different_seed_differs() {
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let grads = Array1::from_vec(vec![0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]);

        let mut a = make_optimizer(1);
        let mut b = make_optimizer(2);
        let out_a = a.step(&params, &grads).expect("a step");
        let out_b = b.step(&params, &grads).expect("b step");

        let diff: f64 = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(
            diff > 0.0,
            "different seeds should produce different updates"
        );
    }

    #[test]
    fn test_ntm_addressing_is_distribution_every_step() {
        // After sharpening, the addressing weights must be a valid probability
        // distribution (non-negative, sum ≈ 1) at every step.
        let mut opt = make_optimizer(17);
        let params = Array1::from_shape_fn(24, |i| 0.5 + 0.1 * (i as f64));
        let grads = Array1::from_shape_fn(24, |i| 0.3 - 0.02 * (i as f64));

        for _ in 0..25 {
            opt.step(&params, &grads).expect("step should succeed");
            assert!(
                is_distribution(opt.last_address_weights()),
                "addressing weights must form a valid probability distribution"
            );
            assert_eq!(
                opt.last_address_weights().len(),
                opt.num_locations(),
                "weights span all memory locations"
            );
        }
    }

    #[test]
    fn test_ntm_content_addressing_is_distribution() {
        let memory = Array2::from_shape_fn((6, 4), |(i, j)| 0.1 * (i as f64) - 0.05 * (j as f64));
        let key = Array1::from_vec(vec![0.2, -0.1, 0.3, 0.05]);
        let w_c = content_addressing(&memory, &key, 2.0);
        assert!(
            is_distribution(&w_c),
            "content weights must be a distribution"
        );
    }

    #[test]
    fn test_ntm_content_addressing_prefers_matching_row() {
        // A key equal to a memory row should give that row the largest weight when
        // the key strength is high.
        let memory = Array2::from_shape_fn((4, 3), |(i, j)| (i as f64) + 0.1 * (j as f64));
        let key = memory.row(2).to_owned();
        let w_c = content_addressing(&memory, &key, 30.0);
        let best = w_c
            .iter()
            .enumerate()
            .fold(
                (0usize, f64::MIN),
                |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                },
            )
            .0;
        assert_eq!(
            best, 2,
            "content addressing should focus on the matching row"
        );
    }

    #[test]
    fn test_ntm_interpolation_blends() {
        let content = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let prev = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
        let blended = interpolate(&content, &prev, 0.25);
        assert!((blended[0] - 0.25).abs() < 1e-12);
        assert!((blended[3] - 0.75).abs() < 1e-12);
        assert!(
            is_distribution(&blended),
            "blend of distributions is a distribution"
        );
    }

    #[test]
    fn test_ntm_circular_shift_rotates() {
        // A shift distribution focused on +1 must rotate the weights by one slot.
        let offsets = build_shift_offsets(1); // [-1, 0, 1]
        let gated = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let shift_right = Array1::from_vec(vec![0.0, 0.0, 1.0]); // offset +1
        let shifted = circular_shift(&gated, &shift_right, &offsets);
        assert!(
            (shifted[1] - 1.0).abs() < 1e-12,
            "mass should move to index 1"
        );
        assert!(is_distribution(&shifted));

        let shift_left = Array1::from_vec(vec![1.0, 0.0, 0.0]); // offset -1
        let shifted_left = circular_shift(&gated, &shift_left, &offsets);
        assert!(
            (shifted_left[3] - 1.0).abs() < 1e-12,
            "leftward shift wraps to the last index"
        );
    }

    #[test]
    fn test_ntm_sharpening_concentrates_distribution() {
        // Larger gamma must reduce the entropy of the distribution.
        let base = softmax(&Array1::from_vec(vec![0.4, 0.1, 0.3, 0.2, 0.0]));
        let mild = sharpen(&base, 1.0);
        let sharp = sharpen(&base, 8.0);
        assert!(is_distribution(&mild));
        assert!(is_distribution(&sharp));
        assert!(
            entropy(&sharp) < entropy(&mild),
            "sharpening with large gamma must decrease entropy: {} !< {}",
            entropy(&sharp),
            entropy(&mild)
        );
    }

    #[test]
    fn test_ntm_write_then_read_recovers_content() {
        // Writing with a one-hot weight and full erase, then reading with the same
        // one-hot weight, must recover the written content.
        let mut memory =
            Array2::from_shape_fn((5, 3), |(i, j)| 0.3 * (i as f64) - 0.1 * (j as f64));
        let mut weights = Array1::<f64>::zeros(5);
        weights[2] = 1.0; // fully focus on location 2
        let erase = Array1::from_elem(3, 1.0); // erase everything at that location
        let content = Array1::from_vec(vec![0.7, -0.4, 0.9]);

        write_memory(&mut memory, &weights, &erase, &content);
        let read = read_memory(&memory, &weights);

        for (r, c) in read.iter().zip(content.iter()) {
            assert!(
                (r - c).abs() < 1e-12,
                "read must recover the written content"
            );
        }
    }

    #[test]
    fn test_ntm_write_leaves_other_locations_unchanged() {
        let mut memory = Array2::from_shape_fn((4, 2), |(i, j)| (i as f64) + (j as f64));
        let original = memory.clone();
        let mut weights = Array1::<f64>::zeros(4);
        weights[1] = 1.0;
        let erase = Array1::from_elem(2, 1.0);
        let add = Array1::from_vec(vec![5.0, 6.0]);

        write_memory(&mut memory, &weights, &erase, &add);

        for i in [0usize, 2, 3] {
            for j in 0..2 {
                assert!(
                    (memory[[i, j]] - original[[i, j]]).abs() < 1e-12,
                    "untouched locations must be unchanged"
                );
            }
        }
        assert!((memory[[1, 0]] - 5.0).abs() < 1e-12);
        assert!((memory[[1, 1]] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_ntm_convergence_quadratic() {
        // f(x) = ||x||^2, grad = 2x. Repeated steps must reduce ||x|| substantially.
        let mut opt = make_optimizer(42);
        let mut x = Array1::from_vec(vec![
            3.0, -3.0, 2.5, -2.5, 4.0, -1.0, 1.5, -2.0, 3.5, -0.5, 2.0, -1.5, 1.0, -3.0, 2.0, -2.5,
        ]);
        let initial = l2(&x);

        for _ in 0..400 {
            let grad = x.mapv(|v| 2.0 * v);
            x = opt.step(&x, &grad).expect("step should succeed");
        }

        let final_norm = l2(&x);
        assert!(
            final_norm < 0.5 * initial,
            "norm should reduce substantially over many iterations: {initial} -> {final_norm}"
        );
    }

    #[test]
    fn test_ntm_memory_evolves_across_steps() {
        let mut opt = make_optimizer(7);
        let params = Array1::from_shape_fn(12, |i| 0.5 + 0.1 * (i as f64));
        let grads = Array1::from_shape_fn(12, |i| 0.3 - 0.02 * (i as f64));

        let _ = opt.step(&params, &grads).expect("first step");
        let memory_after_first = opt.memory().clone();

        let _ = opt.step(&params, &grads).expect("second step");
        let memory_after_second = opt.memory().clone();

        let changed: f64 = memory_after_first
            .iter()
            .zip(memory_after_second.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(changed > 0.0, "external memory should evolve across steps");
        assert_eq!(opt.get_state().step_count, 2);
    }

    #[test]
    fn test_ntm_num_slots_clamped() {
        let cfg = NtmOptimizerConfig {
            num_slots: 1000,
            seed: 5,
            ..Default::default()
        };
        let mut opt = NtmOptimizer::new(cfg).expect("config valid");
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let grads = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        let out = opt.step(&params, &grads).expect("step should succeed");
        assert_eq!(out.len(), 4);
        assert_eq!(
            opt.num_slots(),
            4,
            "num_slots must clamp to the parameter count"
        );
    }

    #[test]
    fn test_ntm_single_parameter() {
        let mut opt = make_optimizer(13);
        let params = Array1::from_vec(vec![5.0]);
        let grads = Array1::from_vec(vec![2.0 * 5.0]);
        let out = opt.step(&params, &grads).expect("step should succeed");
        assert_eq!(out.len(), 1);
        assert_eq!(opt.num_slots(), 1);
        assert!(
            out[0].abs() < params[0].abs(),
            "single param should move toward 0"
        );
    }

    #[test]
    fn test_ntm_trait_methods() {
        let mut opt = make_optimizer(11);
        assert_eq!(opt.name(), "NtmOptimizer");

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let grads = Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1]);
        let _ = opt.step(&params, &grads).expect("step should succeed");

        let state = opt.get_state();
        assert_eq!(state.step_count, 1);
        assert!(state.grad_norm_ema > 0.0);
        assert!(state.current_lr > 0.0);

        opt.set_learning_rate(0.5);
        assert!((opt.get_learning_rate() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_ntm_dimension_mismatch() {
        let mut opt = make_optimizer(3);
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        assert!(opt.step(&params, &grads).is_err());
    }

    #[test]
    fn test_ntm_empty_params_error() {
        let mut opt = make_optimizer(3);
        let params = Array1::<f64>::zeros(0);
        let grads = Array1::<f64>::zeros(0);
        assert!(opt.step(&params, &grads).is_err());
    }

    #[test]
    fn test_ntm_invalid_config() {
        assert!(NtmOptimizer::<f64>::new(NtmOptimizerConfig {
            num_slots: 0,
            ..Default::default()
        })
        .is_err());
        assert!(NtmOptimizer::<f64>::new(NtmOptimizerConfig {
            num_locations: 0,
            ..Default::default()
        })
        .is_err());
        assert!(NtmOptimizer::<f64>::new(NtmOptimizerConfig {
            mem_width: 0,
            ..Default::default()
        })
        .is_err());
        assert!(NtmOptimizer::<f64>::new(NtmOptimizerConfig {
            hidden_dim: 0,
            ..Default::default()
        })
        .is_err());
        assert!(NtmOptimizer::<f64>::new(NtmOptimizerConfig {
            shift_range: 0,
            ..Default::default()
        })
        .is_err());
        assert!(NtmOptimizer::<f64>::new(NtmOptimizerConfig {
            base_lr: 0.0,
            ..Default::default()
        })
        .is_err());
        assert!(NtmOptimizer::<f64>::new(NtmOptimizerConfig {
            momentum_decay: 1.0,
            ..Default::default()
        })
        .is_err());
    }

    #[test]
    fn test_ntm_addressing_pipeline_distribution() {
        // The full addressing pipeline must yield a valid distribution regardless
        // of the head parameters.
        let memory = Array2::from_shape_fn((8, 4), |(i, j)| 0.05 * (i as f64) + 0.1 * (j as f64));
        let prev = Array1::from_elem(8, 1.0 / 8.0);
        let offsets = build_shift_offsets(2);
        let heads = HeadParams {
            key: Array1::from_vec(vec![0.3, -0.2, 0.1, 0.4]),
            beta: 3.5,
            gate: 0.6,
            shift: softmax(&Array1::from_vec(vec![0.2, 0.5, 0.1, 0.0, 0.3])),
            gamma: 4.0,
            erase: Array1::from_elem(4, 0.5),
            add: Array1::from_vec(vec![0.1, -0.1, 0.2, 0.0]),
        };
        let weights = address(&memory, &heads, &prev, &offsets);
        assert_eq!(weights.len(), 8);
        assert!(
            is_distribution(&weights),
            "pipeline output must be a distribution"
        );
    }
}
