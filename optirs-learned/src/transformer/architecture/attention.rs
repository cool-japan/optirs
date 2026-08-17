use std::fmt::Debug;
// Multi-head attention mechanisms for transformer-based optimization
//
// This module implements the attention mechanisms used in the transformer optimizer,
// including multi-head attention, relative position bias, and rotary position embeddings.

use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayViewMut2};
use scirs2_core::numeric::Float;
use scirs2_core::random::{Random, Rng as SCRRng};

use super::super::TransformerOptimizerConfig;
use super::positional_encoding::PositionalEncodingType;
use crate::error::{OptimError, Result};

/// Attention optimization strategies
#[derive(Debug, Clone, Copy, Default)]
pub enum AttentionOptimization {
    /// Standard full attention
    #[default]
    Full,
    /// Sparse attention patterns
    Sparse,
    /// Linear attention approximation
    Linear,
    /// Local attention windows
    Local,
    /// Hierarchical attention
    Hierarchical,
    /// Adaptive attention sparsity
    Adaptive,
}

/// Multi-head attention mechanism for transformer optimizer
#[derive(Debug, Clone)]
pub struct MultiHeadAttention<T: Float + Debug + Send + Sync + 'static> {
    /// Query, Key, Value projection weights
    wq: Array2<T>,
    wk: Array2<T>,
    wv: Array2<T>,

    /// Output projection weights
    wo: Array2<T>,

    /// Number of attention heads
    numheads: usize,

    /// Head dimension
    head_dim: usize,

    /// Model dimension
    modeldim: usize,

    /// Attention optimization strategy
    optimization: AttentionOptimization,

    /// Relative position bias (if enabled)
    relative_bias: Option<RelativePositionBias<T>>,

    /// Attention weights (post-softmax) from the last forward pass
    attentionscores: Option<Array3<T>>,

    /// RoPE embeddings (if enabled)
    rope_embeddings: Option<RoPEEmbeddings<T>>,

    /// ALiBi slopes, one per head (if the ALiBi encoding is selected)
    alibi_slopes: Option<Array1<T>>,

    /// Whether future positions are masked out (autoregressive attention)
    causal: bool,
}

/// Relative position bias for attention
#[derive(Debug, Clone)]
pub struct RelativePositionBias<T: Float + Debug + Send + Sync + 'static> {
    /// Bias table indexed by `[relative_distance, head]`
    bias_table: Array2<T>,

    /// Maximum relative distance
    max_distance: usize,
}

/// Rotary Position Embedding (RoPE)
#[derive(Debug, Clone)]
pub struct RoPEEmbeddings<T: Float + Debug + Send + Sync + 'static> {
    /// Cosine values
    cos_cached: Array2<T>,

    /// Sine values
    sin_cached: Array2<T>,

    /// Maximum sequence length
    max_seqlen: usize,

    /// Dimension
    dim: usize,
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
    > MultiHeadAttention<T>
{
    /// Create a new multi-head attention block from the optimizer configuration.
    pub fn new(config: &TransformerOptimizerConfig) -> Result<Self> {
        let modeldim = config.modeldim;
        let numheads = config.numheads;

        if numheads == 0 || modeldim == 0 {
            return Err(OptimError::InvalidConfig(
                "Model dimension and number of heads must be positive".to_string(),
            ));
        }
        if !modeldim.is_multiple_of(numheads) {
            return Err(OptimError::InvalidConfig(format!(
                "Model dimension {modeldim} must be divisible by the number of heads {numheads}"
            )));
        }
        let head_dim = modeldim / numheads;

        let mut rng = scirs2_core::random::thread_rng();

        // Xavier/Glorot uniform: limit = sqrt(6 / (fan_in + fan_out)). Each of
        // the four projections maps modeldim -> modeldim, so fan_in = fan_out =
        // modeldim and the sum is 2 * modeldim.
        let bound = (6.0 / (2 * modeldim) as f64).sqrt();
        let mut init = |rows: usize, cols: usize| {
            let mut weights = Array2::zeros((rows, cols));
            for elem in weights.iter_mut() {
                *elem =
                    scirs2_core::numeric::NumCast::from((rng.random::<f64>() - 0.5) * 2.0 * bound)
                        .unwrap_or_else(|| T::zero());
            }
            weights
        };

        let wq = init(modeldim, modeldim);
        let wk = init(modeldim, modeldim);
        let wv = init(modeldim, modeldim);
        let wo = init(modeldim, modeldim);

        let relative_bias = if config.relative_position_bias {
            Some(RelativePositionBias::new(
                config.max_sequence_length,
                numheads,
            )?)
        } else {
            None
        };

        let rope_embeddings =
            if config.use_rope || config.pos_encoding_type == PositionalEncodingType::Rotary {
                Some(RoPEEmbeddings::new(config.max_sequence_length, head_dim)?)
            } else {
                None
            };

        let alibi_slopes = if config.pos_encoding_type == PositionalEncodingType::ALiBi {
            let mut slopes = Array1::zeros(numheads);
            for h in 0..numheads {
                slopes[h] = scirs2_core::numeric::NumCast::from(
                    2.0_f64.powf(-8.0 * (h + 1) as f64 / numheads as f64),
                )
                .unwrap_or_else(|| T::zero());
            }
            Some(slopes)
        } else {
            None
        };

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            numheads,
            head_dim,
            modeldim,
            optimization: config.attention_optimization,
            relative_bias,
            attentionscores: None,
            rope_embeddings,
            alibi_slopes,
            causal: config.causal_attention,
        })
    }

    /// Forward pass. `query`, `key` and `value` are `(seq_len, modeldim)`.
    pub fn forward(
        &mut self,
        query: &Array2<T>,
        key: &Array2<T>,
        value: &Array2<T>,
    ) -> Result<Array2<T>> {
        let (seq_len, modeldim) = query.dim();

        if modeldim != self.modeldim {
            return Err(OptimError::InvalidConfig(format!(
                "Model dimension {} doesn't match expected {}",
                modeldim, self.modeldim
            )));
        }
        if key.dim() != query.dim() || value.dim() != query.dim() {
            return Err(OptimError::InvalidConfig(format!(
                "Query {:?}, key {:?} and value {:?} must have identical shapes",
                query.dim(),
                key.dim(),
                value.dim()
            )));
        }
        if seq_len == 0 {
            return Ok(Array2::zeros((0, self.modeldim)));
        }

        // Project to Q, K, V
        let q = self.linear_transform(query, &self.wq);
        let k = self.linear_transform(key, &self.wk);
        let v = self.linear_transform(value, &self.wv);

        // Reshape for multi-head attention
        let q_heads = self.reshape_for_heads(&q);
        let k_heads = self.reshape_for_heads(&k);
        let v_heads = self.reshape_for_heads(&v);

        // Compute attention
        let attention_output = self.compute_attention(&q_heads, &k_heads, &v_heads)?;

        // Reshape back and apply output projection
        let concat_output = self.reshape_from_heads(&attention_output);
        let final_output = self.linear_transform(&concat_output, &self.wo);

        Ok(final_output)
    }

    /// `input @ weights^T`, computed with a single BLAS-style matrix product.
    fn linear_transform(&self, input: &Array2<T>, weights: &Array2<T>) -> Array2<T> {
        input.dot(&weights.t())
    }

    fn reshape_for_heads(&self, input: &Array2<T>) -> Array3<T> {
        let (seq_len, _) = input.dim();
        let mut reshaped = Array3::zeros((self.numheads, seq_len, self.head_dim));

        for h in 0..self.numheads {
            for s_idx in 0..seq_len {
                for d in 0..self.head_dim {
                    reshaped[[h, s_idx, d]] = input[[s_idx, h * self.head_dim + d]];
                }
            }
        }

        reshaped
    }

    fn reshape_from_heads(&self, input: &Array3<T>) -> Array2<T> {
        let (_numheads, seq_len, _head_dim) = input.dim();
        let mut reshaped = Array2::zeros((seq_len, self.modeldim));

        for h in 0..self.numheads {
            for s_idx in 0..seq_len {
                for d in 0..self.head_dim {
                    reshaped[[s_idx, h * self.head_dim + d]] = input[[h, s_idx, d]];
                }
            }
        }

        reshaped
    }

    fn compute_attention(
        &mut self,
        q: &Array3<T>,
        k: &Array3<T>,
        v: &Array3<T>,
    ) -> Result<Array3<T>> {
        let (_numheads, seq_len, head_dim) = q.dim();
        let scale: T = scirs2_core::numeric::NumCast::from(1.0 / (head_dim as f64).sqrt())
            .unwrap_or_else(|| T::one());
        let positions: Vec<usize> = (0..seq_len).collect();

        let mut attention_output = Array3::zeros((self.numheads, seq_len, self.head_dim));
        let mut attention_weights = Array3::zeros((self.numheads, seq_len, seq_len));

        for h in 0..self.numheads {
            let mut q_head = q.slice(s![h, .., ..]).to_owned();
            let mut k_head = k.slice(s![h, .., ..]).to_owned();

            // Rotary position embeddings act on Q and K inside the attention
            // computation - this is the only place RoPE can be applied.
            if let Some(ref rope) = self.rope_embeddings {
                rope.apply_rope(&mut q_head, &positions)?;
                rope.apply_rope(&mut k_head, &positions)?;
            }

            // Scaled dot-product scores for this head.
            let mut scores = q_head.dot(&k_head.t()) * scale;

            // Learned relative position bias, per head.
            if let Some(ref bias) = self.relative_bias {
                bias.apply_bias(&mut scores.view_mut(), h)?;
            }

            // ALiBi: linear distance penalty with a head-specific slope.
            if let Some(ref slopes) = self.alibi_slopes {
                let slope = slopes[h.min(slopes.len().saturating_sub(1))];
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let distance: T =
                            scirs2_core::numeric::NumCast::from((i as i64 - j as i64).abs() as f64)
                                .unwrap_or_else(|| T::zero());
                        scores[[i, j]] = scores[[i, j]] - slope * distance;
                    }
                }
            }

            // Causal masking: position i may not attend to any j > i.
            if self.causal {
                let neg_inf = T::neg_infinity();
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        scores[[i, j]] = neg_inf;
                    }
                }
            }

            Self::apply_softmax(&mut scores.view_mut());

            // Apply attention to values.
            let v_head = v.slice(s![h, .., ..]);
            let head_output = scores.dot(&v_head);

            attention_output
                .slice_mut(s![h, .., ..])
                .assign(&head_output);
            attention_weights.slice_mut(s![h, .., ..]).assign(&scores);
        }

        // Cache post-softmax attention weights for analysis.
        self.attentionscores = Some(attention_weights);

        Ok(attention_output)
    }

    /// Row-wise softmax that tolerates fully masked ("padding") rows.
    ///
    /// A row whose scores are all `-inf` would otherwise produce `0/0 = NaN`;
    /// such rows are emitted as all-zero attention instead.
    fn apply_softmax(scores: &mut ArrayViewMut2<T>) {
        let (rows, cols) = scores.dim();

        for i in 0..rows {
            let mut max_val = T::neg_infinity();
            for j in 0..cols {
                if scores[[i, j]] > max_val {
                    max_val = scores[[i, j]];
                }
            }

            if !max_val.is_finite() {
                // Fully masked row: no position is attendable.
                for j in 0..cols {
                    scores[[i, j]] = T::zero();
                }
                continue;
            }

            let mut sum = T::zero();
            for j in 0..cols {
                let exp_val = (scores[[i, j]] - max_val).exp();
                scores[[i, j]] = exp_val;
                sum = sum + exp_val;
            }

            if sum > T::zero() {
                for j in 0..cols {
                    scores[[i, j]] = scores[[i, j]] / sum;
                }
            } else {
                for j in 0..cols {
                    scores[[i, j]] = T::zero();
                }
            }
        }
    }

    /// Get attention patterns (post-softmax weights) for analysis
    pub fn get_attention_patterns(&self) -> Option<&Array3<T>> {
        self.attentionscores.as_ref()
    }

    /// Get number of attention heads
    pub fn num_heads(&self) -> usize {
        self.numheads
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get the model dimension
    pub fn model_dim(&self) -> usize {
        self.modeldim
    }

    /// Get the configured attention optimization strategy
    pub fn optimization(&self) -> AttentionOptimization {
        self.optimization
    }

    /// Whether causal masking is enabled
    pub fn is_causal(&self) -> bool {
        self.causal
    }

    /// Enable or disable causal masking
    pub fn set_causal(&mut self, causal: bool) {
        self.causal = causal;
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> RelativePositionBias<T> {
    /// Create a bias table covering distances `0 ..= max_distance - 1` for every head.
    pub fn new(max_distance: usize, num_heads: usize) -> Result<Self> {
        if max_distance == 0 || num_heads == 0 {
            return Err(OptimError::InvalidConfig(
                "Relative position bias needs a positive distance and head count".to_string(),
            ));
        }

        let mut rng = scirs2_core::random::thread_rng();
        let table_size = 2 * max_distance - 1;

        let mut bias_table = Array2::zeros((table_size, num_heads));
        let bound = 0.02;

        for elem in bias_table.iter_mut() {
            *elem = scirs2_core::numeric::NumCast::from((rng.random::<f64>() - 0.5) * 2.0 * bound)
                .unwrap_or_else(|| T::zero());
        }

        Ok(Self {
            bias_table,
            max_distance,
        })
    }

    /// Add the learned bias for `head` to a `(seq_len, seq_len)` score matrix.
    pub fn apply_bias(&self, scores: &mut ArrayViewMut2<T>, head: usize) -> Result<()> {
        let (rows, cols) = scores.dim();
        let heads = self.bias_table.ncols();
        if head >= heads {
            return Err(OptimError::InvalidConfig(format!(
                "Head index {head} exceeds the {heads} heads in the relative bias table"
            )));
        }

        for i in 0..rows {
            for j in 0..cols {
                let rel_pos = (i as i64 - j as i64).unsigned_abs() as usize;
                let bias_idx = rel_pos.min(self.max_distance - 1);
                scores[[i, j]] = scores[[i, j]] + self.bias_table[[bias_idx, head]];
            }
        }

        Ok(())
    }

    /// Number of heads covered by the table
    pub fn num_heads(&self) -> usize {
        self.bias_table.ncols()
    }

    /// Maximum relative distance
    pub fn max_distance(&self) -> usize {
        self.max_distance
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> RoPEEmbeddings<T> {
    /// Precompute the rotation table for `max_seqlen` positions of width `dim`.
    pub fn new(max_seqlen: usize, dim: usize) -> Result<Self> {
        if !dim.is_multiple_of(2) {
            return Err(OptimError::InvalidConfig(
                "RoPE dimension must be even".to_string(),
            ));
        }

        let mut cos_cached = Array2::zeros((max_seqlen, dim));
        let mut sin_cached = Array2::zeros((max_seqlen, dim));

        let base = 10000.0_f64;

        for pos in 0..max_seqlen {
            for i in (0..dim).step_by(2) {
                // Both halves of the pair (i, i+1) share the frequency.
                let theta = pos as f64 / base.powf((i as f64) / (dim as f64));
                let cos_val: T =
                    scirs2_core::numeric::NumCast::from(theta.cos()).unwrap_or_else(|| T::one());
                let sin_val: T =
                    scirs2_core::numeric::NumCast::from(theta.sin()).unwrap_or_else(|| T::zero());
                cos_cached[[pos, i]] = cos_val;
                cos_cached[[pos, i + 1]] = cos_val;
                sin_cached[[pos, i]] = sin_val;
                sin_cached[[pos, i + 1]] = sin_val;
            }
        }

        Ok(Self {
            cos_cached,
            sin_cached,
            max_seqlen,
            dim,
        })
    }

    /// Rotate each `(2k, 2k+1)` coordinate pair of `x` by the angle of its position.
    pub fn apply_rope(&self, x: &mut Array2<T>, positions: &[usize]) -> Result<()> {
        let (rows, cols) = x.dim();
        if cols != self.dim {
            return Err(OptimError::InvalidConfig(format!(
                "RoPE expects width {} but received {}",
                self.dim, cols
            )));
        }
        if positions.len() != rows {
            return Err(OptimError::InvalidConfig(format!(
                "RoPE received {} positions for {} rows",
                positions.len(),
                rows
            )));
        }

        for (seq_idx, &pos) in positions.iter().enumerate() {
            if pos >= self.max_seqlen {
                return Err(OptimError::InvalidConfig(format!(
                    "Position {} exceeds the maximum sequence length {}",
                    pos, self.max_seqlen
                )));
            }

            for i in (0..self.dim).step_by(2) {
                let x_even = x[[seq_idx, i]];
                let x_odd = x[[seq_idx, i + 1]];

                let cos_val = self.cos_cached[[pos, i]];
                let sin_val = self.sin_cached[[pos, i]];

                x[[seq_idx, i]] = x_even * cos_val - x_odd * sin_val;
                x[[seq_idx, i + 1]] = x_even * sin_val + x_odd * cos_val;
            }
        }

        Ok(())
    }

    /// Maximum supported sequence length
    pub fn max_sequence_length(&self) -> usize {
        self.max_seqlen
    }

    /// Rotation width
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerOptimizerConfig;

    fn config() -> TransformerOptimizerConfig {
        TransformerOptimizerConfig {
            modeldim: 8,
            numheads: 2,
            max_sequence_length: 16,
            ..Default::default()
        }
    }

    #[test]
    fn causal_mask_blocks_future_positions() {
        let mut cfg = config();
        cfg.causal_attention = true;
        let mut attention = MultiHeadAttention::<f64>::new(&cfg).expect("attention creation");

        let input = Array2::<f64>::from_shape_fn((4, 8), |(i, j)| (i * 8 + j) as f64 * 0.01);
        let _ = attention
            .forward(&input, &input, &input)
            .expect("attention forward");

        let weights = attention
            .get_attention_patterns()
            .expect("attention weights cached");
        for h in 0..2 {
            for i in 0..4 {
                for j in (i + 1)..4 {
                    assert_eq!(
                        weights[[h, i, j]],
                        0.0,
                        "position {i} attended to future position {j}"
                    );
                }
            }
            // Rows still form a probability distribution over the visible prefix.
            for i in 0..4 {
                let row_sum: f64 = (0..4).map(|j| weights[[h, i, j]]).sum();
                assert!((row_sum - 1.0).abs() < 1e-9, "row {i} sums to {row_sum}");
            }
        }
    }

    #[test]
    fn non_causal_attention_sees_the_whole_sequence() {
        let mut cfg = config();
        cfg.causal_attention = false;
        let mut attention = MultiHeadAttention::<f64>::new(&cfg).expect("attention creation");

        let input = Array2::<f64>::from_shape_fn((4, 8), |(i, j)| (i * 8 + j) as f64 * 0.01);
        let _ = attention
            .forward(&input, &input, &input)
            .expect("attention forward");
        let weights = attention
            .get_attention_patterns()
            .expect("attention weights cached");
        assert!(weights[[0, 0, 3]] > 0.0);
    }

    #[test]
    fn mismatched_head_count_is_rejected() {
        let mut cfg = config();
        cfg.numheads = 3; // 8 is not divisible by 3
        assert!(MultiHeadAttention::<f64>::new(&cfg).is_err());
    }

    #[test]
    fn rope_is_a_rotation() {
        let rope = RoPEEmbeddings::<f64>::new(16, 4).expect("rope creation");
        let mut x =
            Array2::<f64>::from_shape_vec((1, 4), vec![1.0, 0.0, 0.0, 1.0]).expect("valid shape");
        let before: f64 = x.iter().map(|v| v * v).sum();
        rope.apply_rope(&mut x, &[3]).expect("rope application");
        let after: f64 = x.iter().map(|v| v * v).sum();
        assert!(
            (before - after).abs() < 1e-12,
            "rotation must preserve norm"
        );

        // Position 0 is the identity rotation.
        let mut y =
            Array2::<f64>::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).expect("valid shape");
        rope.apply_rope(&mut y, &[0]).expect("rope application");
        assert!((y[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((y[[0, 1]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn rope_changes_attention_scores() {
        let mut cfg = config();
        cfg.causal_attention = false;
        cfg.use_rope = false;
        let mut plain = MultiHeadAttention::<f64>::new(&cfg).expect("attention creation");
        let mut roped = plain.clone();
        roped.rope_embeddings = Some(RoPEEmbeddings::new(16, 4).expect("rope creation"));

        let input = Array2::<f64>::from_shape_fn((4, 8), |(i, j)| ((i + 1) * (j + 1)) as f64 * 0.1);
        let _ = plain
            .forward(&input, &input, &input)
            .expect("plain forward");
        let _ = roped
            .forward(&input, &input, &input)
            .expect("roped forward");

        let a = plain.get_attention_patterns().expect("weights");
        let b = roped.get_attention_patterns().expect("weights");
        let diff: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(diff > 1e-6, "RoPE had no effect on attention");
    }

    #[test]
    fn relative_bias_is_head_specific() {
        let bias = RelativePositionBias::<f64>::new(8, 4).expect("bias creation");
        let mut scores_head0 = Array2::<f64>::zeros((3, 3));
        let mut scores_head3 = Array2::<f64>::zeros((3, 3));
        bias.apply_bias(&mut scores_head0.view_mut(), 0)
            .expect("bias head 0");
        bias.apply_bias(&mut scores_head3.view_mut(), 3)
            .expect("bias head 3");
        assert!(
            scores_head0 != scores_head3,
            "every head received the same bias"
        );
        assert!(bias.apply_bias(&mut scores_head0.view_mut(), 4).is_err());
    }

    #[test]
    fn fully_masked_rows_stay_finite() {
        let mut scores = Array2::<f64>::from_elem((2, 2), f64::NEG_INFINITY);
        MultiHeadAttention::<f64>::apply_softmax(&mut scores.view_mut());
        assert!(scores.iter().all(|v| v.is_finite()), "{scores:?}");
        assert!(scores.iter().all(|&v| v == 0.0));
    }
}
