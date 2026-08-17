use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::{thread_rng, Rng};
use std::fmt::Debug;

/// Gradient compression strategies for communication optimization
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionStrategy {
    /// No compression
    None,
    /// Top-K sparsification (keep only top K largest gradients)
    TopK {
        /// Number of top gradients to keep
        k: usize,
    },
    /// Random-K sparsification (keep K random gradients)
    RandomK {
        /// Number of random gradients to keep
        k: usize,
    },
    /// Threshold-based sparsification (keep gradients above threshold)
    Threshold {
        /// Threshold value for gradient magnitude
        threshold: f64,
    },
    /// Quantization to fewer bits
    Quantization {
        /// Number of bits for quantization
        bits: u8,
    },
    /// Error feedback compression (maintain error state)
    ErrorFeedback {
        /// Base compression strategy to apply
        base_strategy: Box<CompressionStrategy>,
        /// Whether to enable error compensation
        error_compensation: bool,
    },
    /// Gradient clipping before compression
    ClippedCompression {
        /// Base compression strategy to apply after clipping
        base_strategy: Box<CompressionStrategy>,
        /// Value to clip gradients to
        clip_value: f64,
    },
}

/// Read a little-endian `f64` out of a byte slice, returning an honest error
/// instead of panicking on a truncated/corrupted buffer (e.g. from a
/// tampered or short network payload).
fn read_f64_le(bytes: &[u8]) -> Result<f64> {
    let arr: [u8; 8] = bytes.try_into().map_err(|_| {
        OptimError::InvalidConfig(
            "corrupted compressed data: expected 8 bytes for an f64".to_string(),
        )
    })?;
    Ok(f64::from_le_bytes(arr))
}

/// Read a little-endian `u32` out of a byte slice, returning an honest error
/// instead of panicking on a truncated/corrupted buffer.
fn read_u32_le(bytes: &[u8]) -> Result<u32> {
    let arr: [u8; 4] = bytes.try_into().map_err(|_| {
        OptimError::InvalidConfig(
            "corrupted compressed data: expected 4 bytes for a u32".to_string(),
        )
    })?;
    Ok(u32::from_le_bytes(arr))
}

/// Read a little-endian `u16` out of a byte slice, returning an honest error
/// instead of panicking on a truncated/corrupted buffer.
fn read_u16_le(bytes: &[u8]) -> Result<u16> {
    let arr: [u8; 2] = bytes.try_into().map_err(|_| {
        OptimError::InvalidConfig(
            "corrupted compressed data: expected 2 bytes for a u16".to_string(),
        )
    })?;
    Ok(u16::from_le_bytes(arr))
}

/// Compressed gradient representation
#[derive(Debug, Clone)]
pub struct CompressedGradient<A: Float> {
    /// Compressed data
    pub data: Vec<u8>,
    /// Compression metadata
    pub metadata: CompressionMetadata<A>,
    /// Original shape information
    pub shapes: Vec<Vec<usize>>,
}

/// Compression metadata
#[derive(Debug, Clone)]
pub struct CompressionMetadata<A: Float> {
    /// Compression strategy used
    pub strategy: CompressionStrategy,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Number of non-zero elements (for sparse methods)
    pub nnz_count: usize,
    /// Quantization scale factors (for quantization methods)
    pub scale_factors: Vec<A>,
    /// Additional strategy-specific data
    pub extra_data: Vec<u8>,
}

/// Gradient compression engine
#[derive(Debug)]
pub struct GradientCompressor<A: Float, D: Dimension> {
    /// Compression strategy
    strategy: CompressionStrategy,
    /// Error feedback state for error compensation
    error_state: Option<Vec<Array<A, D>>>,
    /// Compression statistics
    stats: CompressionStats,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync>
    GradientCompressor<A, D>
{
    /// Create a new gradient compressor
    pub fn new(strategy: CompressionStrategy) -> Self {
        Self {
            strategy,
            error_state: None,
            stats: CompressionStats::new(),
        }
    }

    /// Initialize error state for error feedback compression
    pub fn initialize_error_state(&mut self, gradientshapes: &[Array<A, D>]) {
        self.error_state = Some(
            gradientshapes
                .iter()
                .map(|g| Array::zeros(g.raw_dim()))
                .collect(),
        );
    }

    /// Compress gradients
    pub fn compress(&mut self, gradients: &[Array<A, D>]) -> Result<CompressedGradient<A>> {
        // Lazily initialize the error-feedback residual the first time a caller
        // selects ErrorFeedback with compensation enabled but never called
        // `initialize_error_state` themselves. Previously this silently
        // degraded to plain (uncompensated) compression with no signal.
        let needs_lazy_init = matches!(
            &self.strategy,
            CompressionStrategy::ErrorFeedback {
                error_compensation: true,
                ..
            }
        ) && self.error_state.is_none();
        if needs_lazy_init {
            self.initialize_error_state(gradients);
        }

        // Apply error feedback if enabled: working = gradient + accumulated residual
        let mut working_gradients: Vec<Array<A, D>> =
            if let Some(ref error_state) = self.error_state {
                gradients
                    .iter()
                    .zip(error_state.iter())
                    .map(|(grad, error)| grad + error)
                    .collect()
            } else {
                gradients.to_vec()
            };

        let (compressed_data, metadata) = match &self.strategy {
            CompressionStrategy::None => self.compress_none(&working_gradients)?,
            CompressionStrategy::TopK { k } => self.compress_topk(&working_gradients, *k)?,
            CompressionStrategy::RandomK { k } => self.compress_randomk(&working_gradients, *k)?,
            CompressionStrategy::Threshold { threshold } => self.compress_threshold(
                &working_gradients,
                A::from(*threshold).ok_or_else(|| {
                    OptimError::InvalidConfig(format!(
                        "threshold {threshold} could not be represented in the parameter type"
                    ))
                })?,
            )?,
            CompressionStrategy::Quantization { bits } => {
                self.compress_quantization(&working_gradients, *bits)?
            }
            CompressionStrategy::ErrorFeedback {
                base_strategy,
                error_compensation,
            } => {
                // Recursively apply base strategy
                let mut temp_compressor = GradientCompressor::new((**base_strategy).clone());
                let compressed = temp_compressor.compress(&working_gradients)?;

                // Honour the `error_compensation` flag: when disabled, no
                // residual should be tracked or applied on future calls.
                if *error_compensation {
                    // EF-SGD residual: e_new = working - decompress(compress(working)).
                    // Using the raw input (`original`) here instead of `working`
                    // (which already folds in the previous residual `e_old`) would
                    // pin the residual at a constant and coordinates that never
                    // individually clear the Top-K threshold would never be sent.
                    let decompressed = temp_compressor.decompress(&compressed)?;
                    if let Some(ref mut error_state) = self.error_state {
                        for ((working, decompressed), error) in working_gradients
                            .iter()
                            .zip(decompressed.iter())
                            .zip(error_state.iter_mut())
                        {
                            *error = working - decompressed;
                        }
                    }
                }

                (compressed.data, compressed.metadata)
            }
            CompressionStrategy::ClippedCompression {
                base_strategy,
                clip_value,
            } => {
                // Clip gradients first
                let clip_val = A::from(*clip_value).ok_or_else(|| {
                    OptimError::InvalidConfig(format!(
                        "clip value {clip_value} could not be represented in the parameter type"
                    ))
                })?;
                for grad in &mut working_gradients {
                    grad.mapv_inplace(|x| {
                        if x > clip_val {
                            clip_val
                        } else if x < -clip_val {
                            -clip_val
                        } else {
                            x
                        }
                    });
                }

                // Apply base compression strategy
                let mut temp_compressor = GradientCompressor::new((**base_strategy).clone());
                let compressed = temp_compressor.compress(&working_gradients)?;
                (compressed.data, compressed.metadata)
            }
        };

        // Collect shape information
        let shapes = gradients.iter().map(|g| g.shape().to_vec()).collect();

        let result = CompressedGradient {
            data: compressed_data,
            metadata,
            shapes,
        };

        // Update statistics
        let original_size = self.calculate_size(gradients);
        let compressed_size = result.data.len();
        self.stats
            .record_compression(original_size, compressed_size);

        Ok(result)
    }

    /// Decompress gradients
    pub fn decompress(&self, compressed: &CompressedGradient<A>) -> Result<Vec<Array<A, D>>> {
        match &compressed.metadata.strategy {
            CompressionStrategy::None => self.decompress_none(compressed),
            CompressionStrategy::TopK { .. } => self.decompress_sparse(compressed),
            CompressionStrategy::RandomK { .. } => self.decompress_sparse(compressed),
            CompressionStrategy::Threshold { .. } => self.decompress_sparse(compressed),
            CompressionStrategy::Quantization { bits } => {
                self.decompress_quantization(compressed, *bits)
            }
            CompressionStrategy::ErrorFeedback { base_strategy, .. } => {
                let temp_compressor = GradientCompressor::new((**base_strategy).clone());
                temp_compressor.decompress(compressed)
            }
            CompressionStrategy::ClippedCompression { base_strategy, .. } => {
                let temp_compressor = GradientCompressor::new((**base_strategy).clone());
                temp_compressor.decompress(compressed)
            }
        }
    }

    /// Compress with no compression (passthrough)
    fn compress_none(
        &self,
        gradients: &[Array<A, D>],
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        let mut data = Vec::new();

        // Simple serialization: store all gradient values sequentially
        for grad in gradients {
            for &val in grad.iter() {
                let bits = val.to_f64().ok_or_else(|| {
                    OptimError::InvalidConfig(
                        "gradient value could not be converted to f64 for serialization"
                            .to_string(),
                    )
                })?;
                data.extend_from_slice(&bits.to_le_bytes());
            }
        }

        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::None,
            compression_ratio: 1.0,
            nnz_count: gradients.iter().map(|g| g.len()).sum(),
            scale_factors: Vec::new(),
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Compress using Top-K sparsification
    fn compress_topk(
        &self,
        gradients: &[Array<A, D>],
        k: usize,
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut total_elements = 0;

        for (grad_idx, grad) in gradients.iter().enumerate() {
            total_elements += grad.len();

            // Collect (signed value, index) pairs once -- capturing the signed
            // value up front avoids an O(n) `.nth()` re-lookup per selected
            // element below (previously O(n*k) per gradient).
            let mut value_indices: Vec<(A, usize)> =
                grad.iter().enumerate().map(|(i, &val)| (val, i)).collect();

            // Sort by absolute value (descending). NaN gradients (e.g. from a
            // diverged run) must not panic a comparator run by `sort_by` --
            // treat incomparable pairs as equal rather than unwrapping.
            value_indices.sort_by(|a, b| {
                b.0.abs()
                    .partial_cmp(&a.0.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Take top k elements
            let k_local = k.min(value_indices.len());
            for &(val, orig_idx) in value_indices.iter().take(k_local) {
                indices.push((grad_idx as u32, orig_idx as u32));
                values.push(val);
            }
        }

        // Serialize sparse representation
        let mut data = Vec::new();

        // Store number of sparse elements
        data.extend_from_slice(&(indices.len() as u32).to_le_bytes());

        // Store indices and values
        for ((grad_idx, elem_idx), value) in indices.iter().zip(values.iter()) {
            data.extend_from_slice(&grad_idx.to_le_bytes());
            data.extend_from_slice(&elem_idx.to_le_bytes());
            let bits = value.to_f64().ok_or_else(|| {
                OptimError::InvalidConfig(
                    "gradient value could not be converted to f64 for serialization".to_string(),
                )
            })?;
            data.extend_from_slice(&bits.to_le_bytes());
        }

        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::TopK { k },
            compression_ratio: data.len() as f64
                / (total_elements.max(1) * std::mem::size_of::<A>()) as f64,
            nnz_count: indices.len(),
            scale_factors: Vec::new(),
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Compress using Random-K sparsification
    fn compress_randomk(
        &self,
        gradients: &[Array<A, D>],
        k: usize,
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut total_elements = 0;
        let mut rng = thread_rng();

        for (grad_idx, grad) in gradients.iter().enumerate() {
            total_elements += grad.len();

            // Random sampling of k indices via a genuine partial Fisher-Yates
            // shuffle. The previous implementation picked a swap index that
            // was a pure function of (grad_idx, i) -- every node selected the
            // identical index set every round (losing Random-K's unbiased-
            // estimator property), and it divided by `grad.len() - i`, which
            // is unreachable-but-fragile when i approaches grad.len().
            let k_local = k.min(grad.len());
            let mut selected_indices: Vec<usize> = (0..grad.len()).collect();
            for i in 0..k_local {
                let remaining = grad.len() - i;
                let swap_idx = i + rng.gen_range(0..remaining);
                selected_indices.swap(i, swap_idx);
            }

            // Flatten once so per-element access below is O(1) instead of the
            // previous O(n) `.nth()` walk (O(n*k) total per gradient).
            let flat: Vec<A> = grad.iter().copied().collect();
            for &idx in selected_indices.iter().take(k_local) {
                indices.push((grad_idx as u32, idx as u32));
                values.push(flat[idx]);
            }
        }

        // Serialize sparse representation (same format as Top-K)
        let mut data = Vec::new();
        data.extend_from_slice(&(indices.len() as u32).to_le_bytes());

        for ((grad_idx, elem_idx), value) in indices.iter().zip(values.iter()) {
            data.extend_from_slice(&grad_idx.to_le_bytes());
            data.extend_from_slice(&elem_idx.to_le_bytes());
            let bits = value.to_f64().ok_or_else(|| {
                OptimError::InvalidConfig(
                    "gradient value could not be converted to f64 for serialization".to_string(),
                )
            })?;
            data.extend_from_slice(&bits.to_le_bytes());
        }

        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::RandomK { k },
            compression_ratio: data.len() as f64
                / (total_elements.max(1) * std::mem::size_of::<A>()) as f64,
            nnz_count: indices.len(),
            scale_factors: Vec::new(),
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Compress using threshold-based sparsification
    fn compress_threshold(
        &self,
        gradients: &[Array<A, D>],
        threshold: A,
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut total_elements = 0;

        for (grad_idx, grad) in gradients.iter().enumerate() {
            total_elements += grad.len();

            for (elem_idx, &val) in grad.iter().enumerate() {
                if val.abs() > threshold {
                    indices.push((grad_idx as u32, elem_idx as u32));
                    values.push(val);
                }
            }
        }

        // Serialize sparse representation
        let mut data = Vec::new();
        data.extend_from_slice(&(indices.len() as u32).to_le_bytes());

        for ((grad_idx, elem_idx), value) in indices.iter().zip(values.iter()) {
            data.extend_from_slice(&grad_idx.to_le_bytes());
            data.extend_from_slice(&elem_idx.to_le_bytes());
            let bits = value.to_f64().ok_or_else(|| {
                OptimError::InvalidConfig(
                    "gradient value could not be converted to f64 for serialization".to_string(),
                )
            })?;
            data.extend_from_slice(&bits.to_le_bytes());
        }

        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::Threshold {
                threshold: threshold.to_f64().ok_or_else(|| {
                    OptimError::InvalidConfig(
                        "threshold could not be converted to f64 for metadata".to_string(),
                    )
                })?,
            },
            compression_ratio: data.len() as f64
                / (total_elements.max(1) * std::mem::size_of::<A>()) as f64,
            nnz_count: indices.len(),
            scale_factors: Vec::new(),
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Compress using quantization
    fn compress_quantization(
        &self,
        gradients: &[Array<A, D>],
        bits: u8,
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        if bits == 0 || bits > 32 {
            return Err(OptimError::InvalidConfig(
                "Quantization bits must be in 1..=32".to_string(),
            ));
        }

        let mut data = Vec::new();
        let mut scale_factors = Vec::new();
        let levels = (1u64 << bits) - 1;
        let levels_a = A::from(levels).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "quantization level count {levels} could not be represented in the parameter type"
            ))
        })?;

        for grad in gradients {
            // Reject non-finite gradients up front: NaN/inf would otherwise
            // corrupt the min/max fold below (whose behaviour on NaN is
            // unspecified) and could drive `normalized` negative or NaN,
            // which used to panic in `to_u64().expect(...)`.
            if grad.iter().any(|v| !v.is_finite()) {
                return Err(OptimError::InvalidConfig(
                    "gradient contains non-finite (NaN/inf) values; cannot quantize".to_string(),
                ));
            }

            // Find min and max values for this gradient
            let min_val = grad.iter().fold(A::infinity(), |acc, &x| acc.min(x));
            let max_val = grad.iter().fold(A::neg_infinity(), |acc, &x| acc.max(x));

            let range = max_val - min_val;
            let scale = if range > A::zero() {
                range / levels_a
            } else {
                A::one()
            };

            scale_factors.push(scale);

            // Quantize each value, clamping into [0, levels] so the u64
            // conversion below can never fail (previously an unclamped
            // negative/NaN `normalized` would panic).
            for &val in grad.iter() {
                let normalized = ((val - min_val) / scale)
                    .max(A::zero())
                    .min(levels_a)
                    .round();
                let quantized = normalized.to_u64().unwrap_or(levels).min(levels) as u32;

                // Store quantized value
                match bits {
                    1..=8 => data.push(quantized as u8),
                    9..=16 => data.extend_from_slice(&(quantized as u16).to_le_bytes()),
                    17..=32 => data.extend_from_slice(&quantized.to_le_bytes()),
                    _ => unreachable!(),
                }
            }

            // Store min value AND scale inline for reconstruction. Carrying
            // both in the byte stream (rather than trusting that the
            // separately-returned `scale_factors[grad_idx]` stays aligned by
            // position) means decompression never depends on a parallel
            // array matching this stream's gradient order.
            let min_bits = min_val.to_f64().ok_or_else(|| {
                OptimError::InvalidConfig(
                    "min value could not be converted to f64 for serialization".to_string(),
                )
            })?;
            let scale_bits = scale.to_f64().ok_or_else(|| {
                OptimError::InvalidConfig(
                    "scale factor could not be converted to f64 for serialization".to_string(),
                )
            })?;
            data.extend_from_slice(&min_bits.to_le_bytes());
            data.extend_from_slice(&scale_bits.to_le_bytes());
        }

        let total_elements: usize = gradients.iter().map(|g| g.len()).sum();
        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::Quantization { bits },
            compression_ratio: data.len() as f64
                / (total_elements.max(1) * std::mem::size_of::<A>()) as f64,
            nnz_count: total_elements,
            scale_factors,
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Decompress uncompressed data
    fn decompress_none(&self, compressed: &CompressedGradient<A>) -> Result<Vec<Array<A, D>>> {
        let mut result = Vec::new();
        let mut data_offset = 0;

        for shape in &compressed.shapes {
            let num_elements: usize = shape.iter().product();
            let mut values = Vec::with_capacity(num_elements);

            for _ in 0..num_elements {
                if data_offset + 8 > compressed.data.len() {
                    return Err(OptimError::InvalidConfig(
                        "Insufficient data for decompression".to_string(),
                    ));
                }

                let value = read_f64_le(&compressed.data[data_offset..data_offset + 8])?;
                values.push(A::from(value).ok_or_else(|| {
                    OptimError::InvalidConfig(
                        "decompressed value could not be represented in the parameter type"
                            .to_string(),
                    )
                })?);
                data_offset += 8;
            }

            // Create a dynamic array first, then convert to the target dimension type
            let dynamic_array = Array::from_shape_vec(shape.as_slice(), values).map_err(|_| {
                OptimError::InvalidConfig("Invalid shape for reconstruction".to_string())
            })?;
            let array = dynamic_array.into_dimensionality::<D>().map_err(|_| {
                OptimError::InvalidConfig("Dimension conversion failed".to_string())
            })?;
            result.push(array);
        }

        Ok(result)
    }

    /// Decompress sparse representation
    fn decompress_sparse(&self, compressed: &CompressedGradient<A>) -> Result<Vec<Array<A, D>>> {
        let mut result = Vec::new();

        // Initialize zero arrays
        for shape in &compressed.shapes {
            let dynamic_array = Array::zeros(shape.as_slice());
            let array = dynamic_array.into_dimensionality::<D>().map_err(|_| {
                OptimError::InvalidConfig("Dimension conversion failed for zero array".to_string())
            })?;
            result.push(array);
        }

        // Read number of sparse elements
        if compressed.data.len() < 4 {
            return Err(OptimError::InvalidConfig(
                "Invalid compressed data format".to_string(),
            ));
        }

        let num_elements = read_u32_le(&compressed.data[0..4])? as usize;
        let mut data_offset = 4;

        // Restore sparse elements
        for _ in 0..num_elements {
            if data_offset + 16 > compressed.data.len() {
                return Err(OptimError::InvalidConfig(
                    "Insufficient data for sparse decompression".to_string(),
                ));
            }

            let grad_idx = read_u32_le(&compressed.data[data_offset..data_offset + 4])? as usize;
            let elem_idx =
                read_u32_le(&compressed.data[data_offset + 4..data_offset + 8])? as usize;
            let value_f64 = read_f64_le(&compressed.data[data_offset + 8..data_offset + 16])?;
            let value = A::from(value_f64).ok_or_else(|| {
                OptimError::InvalidConfig(
                    "decompressed value could not be represented in the parameter type".to_string(),
                )
            })?;

            data_offset += 16;

            if grad_idx >= result.len() {
                return Err(OptimError::InvalidConfig(
                    "Invalid gradient index in compressed data".to_string(),
                ));
            }

            // Write via a flat slice (O(1) indexed access) instead of
            // `.iter_mut().nth(elem_idx)`, which re-walks from the start of
            // the array for every restored element.
            let target = result[grad_idx].as_slice_mut().ok_or_else(|| {
                OptimError::InvalidConfig(
                    "target array is not contiguous; cannot write decompressed element".to_string(),
                )
            })?;
            match target.get_mut(elem_idx) {
                Some(elem) => *elem = value,
                None => {
                    return Err(OptimError::InvalidConfig(
                        "Invalid element index in compressed data".to_string(),
                    ));
                }
            }
        }

        Ok(result)
    }

    /// Decompress quantized data
    fn decompress_quantization(
        &self,
        compressed: &CompressedGradient<A>,
        bits: u8,
    ) -> Result<Vec<Array<A, D>>> {
        let mut result = Vec::new();
        let mut data_offset = 0;

        for shape in compressed.shapes.iter() {
            let num_elements: usize = shape.iter().product();
            let mut values = Vec::with_capacity(num_elements);

            // Read quantized values
            for _ in 0..num_elements {
                let quantized = match bits {
                    1..=8 => {
                        if data_offset >= compressed.data.len() {
                            return Err(OptimError::InvalidConfig(
                                "Insufficient quantized data".to_string(),
                            ));
                        }
                        let val = compressed.data[data_offset] as u32;
                        data_offset += 1;
                        val
                    }
                    9..=16 => {
                        if data_offset + 2 > compressed.data.len() {
                            return Err(OptimError::InvalidConfig(
                                "Insufficient quantized data".to_string(),
                            ));
                        }
                        let val =
                            read_u16_le(&compressed.data[data_offset..data_offset + 2])? as u32;
                        data_offset += 2;
                        val
                    }
                    17..=32 => {
                        if data_offset + 4 > compressed.data.len() {
                            return Err(OptimError::InvalidConfig(
                                "Insufficient quantized data".to_string(),
                            ));
                        }
                        let val = read_u32_le(&compressed.data[data_offset..data_offset + 4])?;
                        data_offset += 4;
                        val
                    }
                    _ => {
                        return Err(OptimError::InvalidConfig(
                            "Invalid quantization bits".to_string(),
                        ))
                    }
                };

                values.push(quantized);
            }

            // Read min value and scale, stored inline by `compress_quantization`
            // right after each gradient's quantized block. Reading them from
            // the stream itself (rather than indexing into the separately
            // carried `metadata.scale_factors` by position) means
            // reconstruction never depends on that parallel array staying
            // aligned with this one.
            if data_offset + 16 > compressed.data.len() {
                return Err(OptimError::InvalidConfig(
                    "Missing min value/scale for quantization".to_string(),
                ));
            }
            let min_val_f64 = read_f64_le(&compressed.data[data_offset..data_offset + 8])?;
            let scale_f64 = read_f64_le(&compressed.data[data_offset + 8..data_offset + 16])?;
            let min_val = A::from(min_val_f64).ok_or_else(|| {
                OptimError::InvalidConfig(
                    "min value could not be represented in the parameter type".to_string(),
                )
            })?;
            let scale = A::from(scale_f64).ok_or_else(|| {
                OptimError::InvalidConfig(
                    "scale factor could not be represented in the parameter type".to_string(),
                )
            })?;
            data_offset += 16;

            // Dequantize values
            let dequantized_values: Vec<A> = values
                .into_iter()
                .map(|q| -> Result<A> {
                    let q_a = A::from(q).ok_or_else(|| {
                        OptimError::InvalidConfig(
                            "quantized value could not be represented in the parameter type"
                                .to_string(),
                        )
                    })?;
                    Ok(min_val + q_a * scale)
                })
                .collect::<Result<Vec<A>>>()?;

            let dynamic_array = Array::from_shape_vec(shape.as_slice(), dequantized_values)
                .map_err(|_| {
                    OptimError::InvalidConfig(
                        "Invalid shape for quantized reconstruction".to_string(),
                    )
                })?;
            let array = dynamic_array.into_dimensionality::<D>().map_err(|_| {
                OptimError::InvalidConfig(
                    "Dimension conversion failed for quantized array".to_string(),
                )
            })?;
            result.push(array);
        }

        Ok(result)
    }

    /// Calculate size of gradients in bytes
    fn calculate_size(&self, gradients: &[Array<A, D>]) -> usize {
        gradients
            .iter()
            .map(|g| g.len() * std::mem::size_of::<A>())
            .sum()
    }

    /// Get compression statistics
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Reset compression statistics
    pub fn reset_stats(&mut self) {
        self.stats = CompressionStats::new();
    }
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Total compressions performed
    pub compressions_count: usize,
    /// Total original bytes
    pub total_original_bytes: usize,
    /// Total compressed bytes
    pub total_compressed_bytes: usize,
    /// Average compression ratio
    pub average_compression_ratio: f64,
    /// Best compression ratio achieved
    pub best_compression_ratio: f64,
    /// Worst compression ratio achieved
    pub worst_compression_ratio: f64,
}

impl CompressionStats {
    /// Create new compression statistics
    pub fn new() -> Self {
        Self {
            compressions_count: 0,
            total_original_bytes: 0,
            total_compressed_bytes: 0,
            average_compression_ratio: 0.0,
            best_compression_ratio: f64::INFINITY,
            worst_compression_ratio: 0.0,
        }
    }

    /// Record a compression operation
    pub fn record_compression(&mut self, original_bytes: usize, compressedbytes: usize) {
        self.compressions_count += 1;
        self.total_original_bytes += original_bytes;
        self.total_compressed_bytes += compressedbytes;

        let ratio = if original_bytes > 0 {
            compressedbytes as f64 / original_bytes as f64
        } else {
            1.0
        };

        self.best_compression_ratio = self.best_compression_ratio.min(ratio);
        self.worst_compression_ratio = self.worst_compression_ratio.max(ratio);

        self.average_compression_ratio = if self.total_original_bytes > 0 {
            self.total_compressed_bytes as f64 / self.total_original_bytes as f64
        } else {
            0.0
        };
    }

    /// Get overall compression ratio
    pub fn overall_compression_ratio(&self) -> f64 {
        self.average_compression_ratio
    }

    /// Get bandwidth savings (as percentage)
    pub fn bandwidth_savings(&self) -> f64 {
        (1.0 - self.average_compression_ratio) * 100.0
    }
}

impl Default for CompressionStats {
    fn default() -> Self {
        Self::new()
    }
}
