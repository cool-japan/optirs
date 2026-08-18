//! The LSTM controller's input feature construction, shared by inference and
//! meta-training.
//!
//! `LSTMOptimizer::prepare_lstm_input` used to build this vector inline. It is
//! extracted here so that the truncated-BPTT meta-trainer
//! ([`super::trainer::MetaTrainer`]) feeds the controller the *exact same*
//! features it will see at inference time — training a different function than
//! the one that ships would make the meta-training meaningless. Both callers now
//! go through [`build_lstm_features`], so they cannot drift apart.
//!
//! The extraction also removed a real panic: the original read the gradient with
//! `gradients.as_slice().expect("unwrap failed")`, which panics for any array
//! that is not in contiguous standard layout (finding F85). This version
//! iterates instead.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};

/// Build the controller's input feature vector.
///
/// Layout, in order:
///
/// 1. every component of `gradients`;
/// 2. for each entry of `recent_gradients` (most recent first, as
///    `HistoryBuffer::get_recent_gradients` returns them), the elementwise
///    difference `gradients − previous` over the overlapping prefix;
/// 3. three summary statistics of `gradients`: L2 norm, mean, standard deviation;
/// 4. `loss_features` (loss change and loss ratio, when available);
/// 5. zero-padding or truncation to exactly `input_features` entries.
///
/// # Errors
/// Returns `Err` when `gradients` is empty or `input_features` is zero — in
/// either case the controller has no usable input.
pub fn build_lstm_features<T>(
    gradients: &Array1<T>,
    recent_gradients: &[&Array1<T>],
    loss_features: Option<&[T]>,
    input_features: usize,
) -> Result<Array1<T>>
where
    T: Float + Debug + Send + Sync + 'static,
{
    if gradients.is_empty() {
        return Err(OptimError::InsufficientData(
            "cannot build LSTM features from an empty gradient".to_string(),
        ));
    }
    if input_features == 0 {
        return Err(OptimError::InvalidConfig(
            "input_features must be greater than 0".to_string(),
        ));
    }

    let mut features: Vec<T> = Vec::with_capacity(input_features);
    features.extend(gradients.iter().copied());

    for previous in recent_gradients {
        features.extend(
            gradients
                .iter()
                .zip(previous.iter())
                .map(|(&g1, &g2)| g1 - g2),
        );
    }

    let count: T = scirs2_core::numeric::NumCast::from(gradients.len()).unwrap_or_else(T::one);
    let grad_norm = gradients
        .iter()
        .map(|&g| g * g)
        .fold(T::zero(), |a, b| a + b)
        .sqrt();
    let grad_mean = gradients.iter().copied().fold(T::zero(), |a, b| a + b) / count;
    let grad_std = (gradients
        .iter()
        .map(|&g| (g - grad_mean) * (g - grad_mean))
        .fold(T::zero(), |a, b| a + b)
        / count)
        .sqrt();
    features.extend([grad_norm, grad_mean, grad_std]);

    if let Some(loss) = loss_features {
        features.extend(loss.iter().copied());
    }

    features.resize(input_features, T::zero());
    Ok(Array1::from_vec(features))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_is_exactly_input_features() {
        let g = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
        for width in [1usize, 3, 6, 64] {
            let f = build_lstm_features(&g, &[], None, width).expect("features");
            assert_eq!(f.len(), width);
        }
    }

    #[test]
    fn statistics_land_right_after_the_gradient() {
        let g = Array1::from_vec(vec![3.0_f64, 4.0]);
        let f = build_lstm_features(&g, &[], None, 5).expect("features");
        assert_eq!(f[0], 3.0);
        assert_eq!(f[1], 4.0);
        assert!((f[2] - 5.0).abs() < 1e-12, "norm {}", f[2]);
        assert!((f[3] - 3.5).abs() < 1e-12, "mean {}", f[3]);
        assert!((f[4] - 0.5).abs() < 1e-12, "std {}", f[4]);
    }

    #[test]
    fn history_contributes_differences() {
        let g = Array1::from_vec(vec![1.0_f64, 1.0]);
        let previous = Array1::from_vec(vec![0.25_f64, 0.5]);
        let f = build_lstm_features(&g, &[&previous], None, 4).expect("features");
        assert_eq!(f[0], 1.0);
        assert_eq!(f[1], 1.0);
        assert!((f[2] - 0.75).abs() < 1e-12, "diff0 {}", f[2]);
        assert!((f[3] - 0.5).abs() < 1e-12, "diff1 {}", f[3]);
    }

    #[test]
    fn loss_features_are_appended() {
        let g = Array1::from_vec(vec![1.0_f64]);
        // [g0, norm, mean, std, loss_change, loss_ratio]
        let f = build_lstm_features(&g, &[], Some(&[-0.5_f64, 0.8]), 6).expect("features");
        assert!((f[4] + 0.5).abs() < 1e-12, "loss change {}", f[4]);
        assert!((f[5] - 0.8).abs() < 1e-12, "loss ratio {}", f[5]);
    }

    #[test]
    fn a_non_contiguous_gradient_does_not_panic() {
        // The old implementation called `as_slice().expect(...)` here, which
        // returns `None` for a strided view and therefore panicked.
        let dense = Array1::from_vec((0..10).map(|i| i as f64).collect::<Vec<_>>());
        let strided = dense.slice(scirs2_core::ndarray::s![..;2]).to_owned();
        let view = dense.slice(scirs2_core::ndarray::s![..;2]);
        assert!(view.as_slice().is_none(), "expected a non-contiguous view");
        let from_view = build_lstm_features(&view.to_owned(), &[], None, 8).expect("features");
        let from_dense = build_lstm_features(&strided, &[], None, 8).expect("features");
        assert_eq!(from_view, from_dense);
    }

    #[test]
    fn rejects_degenerate_inputs() {
        let empty = Array1::<f64>::zeros(0);
        assert!(build_lstm_features(&empty, &[], None, 4).is_err());
        let g = Array1::from_vec(vec![1.0_f64]);
        assert!(build_lstm_features(&g, &[], None, 0).is_err());
    }
}
