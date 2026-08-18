//! Real encoder for the prototypical network (finding F16).
//!
//! Two things were wrong in `few_shot.rs`:
//!
//! 1. `PrototypicalNetwork::new` and `::from_dims` built their `EncoderLayer`
//!    with `Array2::zeros(...)`, so the encoder — had it ever been used — would
//!    have output the zero vector for every input.
//! 2. `encode_task` never used `self.encoder` at all. It computed a truncated
//!    coordinate-wise mean of the *raw* support features, which made the encoder
//!    dead weight and the "embedding" nothing more than the input.
//!
//! This module supplies the pieces that fix both: Xavier initialization and a
//! real [`EncoderNetwork::forward`]. It lives in a child module because
//! `few_shot.rs` is within a few dozen lines of the 2000-line cap.
//!
//! The embedding follows the standard prototypical-network formulation
//! (Snell et al. 2017): each support example is embedded by `f_φ` and the
//! prototype is the *mean of the embeddings*, not the embedding of the mean.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::{ActivationFunction, EncoderLayer, EncoderNetwork, LayerType};
use crate::error::{OptimError, Result};

/// Build a Xavier/Glorot-uniform encoder layer of shape `(input_dim, output_dim)`.
///
/// The limit is `sqrt(6 / (input_dim + output_dim))`. The previous code used
/// `Array2::zeros`, which makes the layer output identically zero and its
/// gradient identically zero — an encoder that can never learn anything.
///
/// # Errors
/// Returns `Err` when either dimension is zero.
pub fn xavier_encoder_layer<T>(
    input_dim: usize,
    output_dim: usize,
    layer_type: LayerType,
) -> Result<EncoderLayer<T>>
where
    T: Float + Debug + Send + Sync + 'static,
{
    if input_dim == 0 || output_dim == 0 {
        return Err(OptimError::InvalidConfig(format!(
            "encoder layer dimensions must be positive, got {input_dim}x{output_dim}"
        )));
    }
    let bound = (6.0 / (input_dim + output_dim) as f64).sqrt();
    let mut rng = scirs2_core::random::thread_rng();
    let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
        let sample = rng.random::<f64>() * 2.0 - 1.0;
        scirs2_core::numeric::NumCast::from(sample * bound).unwrap_or_else(|| T::zero())
    });
    Ok(EncoderLayer {
        weights,
        bias: Array1::zeros(output_dim),
        layer_type,
    })
}

/// Apply an activation elementwise.
///
/// `Swish` uses `x·σ(x)` (never `x·eˣ/(1+eˣ)`, which overflows to `NaN` for
/// large `x`), and `Mish` uses `x·tanh(softplus(x))` with the numerically stable
/// `softplus` for large arguments.
pub fn activate<T>(value: T, activation: ActivationFunction) -> T
where
    T: Float + Debug + Send + Sync + 'static,
{
    let sigmoid = |x: T| T::one() / (T::one() + (-x).exp());
    match activation {
        ActivationFunction::ReLU => {
            if value > T::zero() {
                value
            } else {
                T::zero()
            }
        }
        ActivationFunction::Tanh => value.tanh(),
        ActivationFunction::Sigmoid => sigmoid(value),
        ActivationFunction::Swish => value * sigmoid(value),
        ActivationFunction::GELU => {
            // tanh approximation (Hendrycks & Gimpel 2016).
            let c: T =
                scirs2_core::numeric::NumCast::from(0.797_884_560_802_865_4).unwrap_or_else(T::one);
            let k: T = scirs2_core::numeric::NumCast::from(0.044_715).unwrap_or_else(T::zero);
            let half: T = scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(T::one);
            half * value * (T::one() + (c * (value + k * value * value * value)).tanh())
        }
        ActivationFunction::Mish => {
            // softplus(x) = ln(1 + eˣ); for large x that is just x.
            let threshold: T = scirs2_core::numeric::NumCast::from(20.0).unwrap_or_else(T::one);
            let softplus = if value > threshold {
                value
            } else {
                (T::one() + value.exp()).ln()
            };
            value * softplus.tanh()
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> EncoderNetwork<T> {
    /// Embed a raw feature vector.
    ///
    /// Each layer computes `Wᵗx + b` (the weight matrix is stored
    /// `(input_dim, output_dim)`), and the activation is applied to every layer
    /// **except the last** — a prototypical-network embedding is the final linear
    /// map's output, not a rectified version of it.
    ///
    /// A feature vector that does not match the first layer's input width is
    /// zero-padded or truncated to fit, which is documented rather than silent:
    /// the caller's raw task features have no fixed width, and refusing them
    /// outright would make the encoder unusable on real task data.
    ///
    /// # Errors
    /// Returns `Err` when the network has no layers, or when two consecutive
    /// layers disagree on width.
    pub fn forward(&self, features: &Array1<T>) -> Result<Array1<T>> {
        let first = self.layers.first().ok_or_else(|| {
            OptimError::InvalidConfig("encoder network has no layers".to_string())
        })?;

        let mut activation = Array1::<T>::zeros(first.weights.nrows());
        let copy = features.len().min(activation.len());
        for i in 0..copy {
            activation[i] = features[i];
        }

        let last_index = self.layers.len() - 1;
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.weights.nrows() != activation.len() {
                return Err(OptimError::InvalidConfig(format!(
                    "encoder layer {index} expects {} inputs but received {}",
                    layer.weights.nrows(),
                    activation.len()
                )));
            }
            let out_dim = layer.weights.ncols();
            if layer.bias.len() != out_dim {
                return Err(OptimError::InvalidConfig(format!(
                    "encoder layer {index} has {out_dim} outputs but {} biases",
                    layer.bias.len()
                )));
            }
            let mut next = Array1::<T>::zeros(out_dim);
            for j in 0..out_dim {
                let mut sum = layer.bias[j];
                for i in 0..activation.len() {
                    sum = sum + layer.weights[[i, j]] * activation[i];
                }
                next[j] = if index == last_index {
                    sum
                } else {
                    activate(sum, self.activation)
                };
            }
            activation = next;
        }

        Ok(activation)
    }

    /// Width of the embedding this network produces.
    ///
    /// # Errors
    /// Returns `Err` when the network has no layers.
    pub fn embedding_width(&self) -> Result<usize> {
        self.layers
            .last()
            .map(|l| l.weights.ncols())
            .ok_or_else(|| OptimError::InvalidConfig("encoder network has no layers".to_string()))
    }
}

#[cfg(test)]
impl<T: Float + Debug + Send + Sync + 'static> super::PrototypicalNetwork<T> {
    /// Set one encoder weight. Test-only hook used to prove that the encoder
    /// genuinely participates in `encode_task`.
    pub(crate) fn perturb_encoder_weight_for_test(&mut self, row: usize, col: usize, value: T) {
        if let Some(layer) = self.encoder.layers.first_mut() {
            if row < layer.weights.nrows() && col < layer.weights.ncols() {
                layer.weights[[row, col]] = value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn network(dims: &[(usize, usize)]) -> EncoderNetwork<f64> {
        EncoderNetwork {
            layers: dims
                .iter()
                .map(|&(i, o)| xavier_encoder_layer::<f64>(i, o, LayerType::Linear).expect("layer"))
                .collect(),
            activation: ActivationFunction::ReLU,
        }
    }

    #[test]
    fn xavier_layers_are_not_zero() {
        let layer = xavier_encoder_layer::<f64>(6, 4, LayerType::Linear).expect("layer");
        let magnitude = layer.weights.iter().fold(0.0_f64, |a, &w| a.max(w.abs()));
        assert!(magnitude > 0.0, "encoder weights are all zero");
        let bound = (6.0_f64 / 10.0).sqrt();
        assert!(
            layer.weights.iter().all(|w| w.abs() <= bound + 1e-12),
            "a weight escaped the Xavier bound {bound}"
        );
    }

    #[test]
    fn rejects_zero_dimensions() {
        assert!(xavier_encoder_layer::<f64>(0, 4, LayerType::Linear).is_err());
        assert!(xavier_encoder_layer::<f64>(4, 0, LayerType::Linear).is_err());
    }

    #[test]
    fn forward_produces_the_embedding_width() {
        let net = network(&[(6, 5), (5, 3)]);
        assert_eq!(net.embedding_width().expect("width"), 3);
        let embedded = net
            .forward(&Array1::from_vec(vec![1.0, -0.5, 0.25, 2.0, -1.0, 0.1]))
            .expect("forward");
        assert_eq!(embedded.len(), 3);
        assert!(embedded.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_depends_on_its_input() {
        let net = network(&[(4, 3)]);
        let a = net
            .forward(&Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]))
            .expect("forward on the first one-hot input");
        let b = net
            .forward(&Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]))
            .expect("forward on the second one-hot input");
        let delta = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(
            delta > 1e-9,
            "the encoder ignores its input (delta {delta})"
        );
    }

    #[test]
    fn short_and_long_features_are_padded_and_truncated() {
        let net = network(&[(4, 2)]);
        assert!(net.forward(&Array1::from_vec(vec![1.0, 2.0])).is_ok());
        assert!(net
            .forward(&Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            .is_ok());
        // Padding must be zeros, so a short vector equals the same vector
        // explicitly zero-extended.
        let short = net
            .forward(&Array1::from_vec(vec![1.0, 2.0]))
            .expect("short");
        let padded = net
            .forward(&Array1::from_vec(vec![1.0, 2.0, 0.0, 0.0]))
            .expect("padded");
        assert_eq!(short, padded);
    }

    #[test]
    fn an_empty_network_is_an_error() {
        let empty = EncoderNetwork::<f64> {
            layers: Vec::new(),
            activation: ActivationFunction::ReLU,
        };
        assert!(empty.forward(&Array1::from_vec(vec![1.0])).is_err());
        assert!(empty.embedding_width().is_err());
    }

    /// F16: `PrototypicalNetwork::encode_task` used to compute a truncated
    /// coordinate-wise mean of the **raw** support features and never call
    /// `self.encoder`. It must now embed each example and average the embeddings,
    /// which is observable two ways: the result is not the raw feature mean, and
    /// perturbing an encoder weight changes it.
    #[test]
    fn encode_task_goes_through_the_encoder() {
        use crate::few_shot::{
            DifficultyLevel, DomainCharacteristics, DomainInfo, DomainType, ExampleMetadata,
            PrototypicalNetwork, PrototypicalNetworkConfig, QuerySet, QuerySetStatistics,
            SupportExample, SupportSet, SupportSetStatistics, TaskData, TaskMetadata,
        };
        use std::collections::HashMap;

        let features = [
            Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]),
            Array1::from_vec(vec![-1.0_f64, 0.5, 2.0, -3.0]),
        ];
        let examples: Vec<SupportExample<f64>> = features
            .iter()
            .map(|f| SupportExample {
                features: f.clone(),
                target: 0.0,
                weight: 1.0,
                context: HashMap::new(),
                metadata: ExampleMetadata {
                    source: "test".to_string(),
                    quality_score: 1.0,
                    created_at: std::time::SystemTime::now(),
                },
            })
            .collect();

        let task = TaskData::<f64> {
            task_id: "f16".to_string(),
            support_set: SupportSet {
                examples,
                task_metadata: TaskMetadata {
                    task_name: "f16".to_string(),
                    domain: DomainType::Optimization,
                    difficulty: DifficultyLevel::Easy,
                    created_at: std::time::SystemTime::now(),
                },
                statistics: SupportSetStatistics {
                    mean: Array1::zeros(4),
                    variance: Array1::zeros(4),
                    size: 2,
                    diversity_score: 1.0,
                },
                temporal_order: None,
            },
            query_set: QuerySet {
                examples: Vec::new(),
                statistics: QuerySetStatistics {
                    mean: Array1::zeros(4),
                    variance: Array1::zeros(4),
                    size: 0,
                },
                eval_metrics: Vec::new(),
            },
            task_params: HashMap::new(),
            domain_info: DomainInfo {
                domain_type: DomainType::Optimization,
                characteristics: DomainCharacteristics {
                    input_dim: 4,
                    output_dim: 1,
                    temporal: false,
                    stochasticity: 0.0,
                    noise_level: 0.0,
                    sparsity: 0.0,
                },
                difficulty_level: DifficultyLevel::Easy,
                constraints: Vec::new(),
            },
        };

        let mut network = PrototypicalNetwork::<f64>::new(PrototypicalNetworkConfig {
            embedding_dim: 3,
            learning_rate: 0.01,
            num_layers: 1,
            hidden_dim: 4,
        })
        .expect("network");

        let encoded = network.encode_task(&task).expect("encode_task");
        assert_eq!(
            encoded.len(),
            3,
            "embedding width must come from the encoder"
        );

        // The raw truncated feature mean the old code returned.
        let raw_mean = [
            (features[0][0] + features[1][0]) / 2.0,
            (features[0][1] + features[1][1]) / 2.0,
            (features[0][2] + features[1][2]) / 2.0,
        ];
        let to_raw = encoded
            .iter()
            .zip(raw_mean.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            to_raw > 1e-9,
            "encode_task still returns the raw feature mean (delta {to_raw})"
        );

        // Changing an encoder weight must change the embedding. Use input row 1:
        // input row 0 carries features 1.0 and -1.0, whose mean is exactly zero,
        // so a weight on that row cannot move the averaged embedding at all.
        network.perturb_encoder_weight_for_test(1, 0, 5.0);
        let after = network.encode_task(&task).expect("encode_task again");
        let delta = encoded
            .iter()
            .zip(after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            delta > 1e-9,
            "the encoder weights do not affect the embedding (delta {delta})"
        );
    }

    #[test]
    fn swish_and_mish_stay_finite_for_large_inputs() {
        for a in [ActivationFunction::Swish, ActivationFunction::Mish] {
            for x in [-1e3_f64, -10.0, 0.0, 10.0, 1e3] {
                let y = activate(x, a);
                assert!(y.is_finite(), "{a:?} at {x} produced {y}");
            }
        }
    }
}
