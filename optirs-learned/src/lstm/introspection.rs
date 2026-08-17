//! Read-only views of the LSTM controller's learned weights.
//!
//! Initialization statistics (finding F64) and gradient-flow diagnostics can only
//! be asserted from outside the crate if the weight matrices are reachable, and
//! `LSTMNetwork`/`LSTMLayer`/`OutputProjection` keep all of theirs private.
//! These accessors hand out immutable borrows only — nothing here lets a caller
//! mutate a weight, so the invariants the forward pass relies on (matching
//! `weights.ncols()` and input width, four gate blocks stacked along the rows)
//! stay under the owning type's control.
//!
//! Kept in its own file so `lstm.rs` stays under the 2000-line cap.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::{AttentionMechanism, LSTMLayer, LSTMNetwork, LSTMOptimizer, OutputProjection};

impl<T: Float + Debug + Send + Sync + 'static> OutputProjection<T> {
    /// The projection matrix, shaped `(output_size, input_size)`.
    pub fn weight_snapshot(&self) -> &Array2<T> {
        &self.weights
    }

    /// The projection bias, length `output_size`.
    pub fn bias_snapshot(&self) -> &Array1<T> {
        &self.bias
    }
}

impl<T: Float + Debug + Send + Sync + 'static> LSTMLayer<T> {
    /// Input-to-hidden weights, shaped `(4 · hidden_size, input_size)`.
    ///
    /// The four gate blocks (input, forget, cell, output) are stacked along the
    /// rows, so the *fan-out* of each gate is `hidden_size`, not `4 ·
    /// hidden_size` — which is what the Glorot limit has to be computed from.
    pub fn weight_ih_snapshot(&self) -> &Array2<T> {
        &self.weight_ih
    }

    /// Hidden-to-hidden weights, shaped `(4 · hidden_size, hidden_size)`.
    pub fn weight_hh_snapshot(&self) -> &Array2<T> {
        &self.weight_hh
    }
}

impl<T: Float + Debug + Send + Sync + 'static> AttentionMechanism<T> {
    /// The four attention projections in `(query, key, value, output)` order,
    /// each shaped `(hidden_size, hidden_size)`.
    pub fn projection_snapshots(&self) -> (&Array2<T>, &Array2<T>, &Array2<T>, &Array2<T>) {
        (
            &self.query_proj,
            &self.key_proj,
            &self.value_proj,
            &self.output_proj,
        )
    }
}

impl<T: Float + Debug + Send + Sync + 'static> LSTMNetwork<T> {
    /// Number of stacked LSTM layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Borrow the `index`-th LSTM layer, or `None` if it does not exist.
    pub fn layer(&self, index: usize) -> Option<&LSTMLayer<T>> {
        self.layers.get(index)
    }

    /// Borrow the output projection.
    pub fn output_projection_snapshot(&self) -> &OutputProjection<T> {
        &self.output_projection
    }

    /// Borrow the attention mechanism, if the configuration enabled one.
    pub fn attention_snapshot(&self) -> Option<&AttentionMechanism<T>> {
        self.attention.as_ref()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> LSTMOptimizer<T> {
    /// `(weight_ih, weight_hh)` of the first LSTM layer, or `None` when the
    /// controller has no layers.
    pub fn first_layer_weight_snapshot(&self) -> Option<(&Array2<T>, &Array2<T>)> {
        self.lstm_network
            .layer(0)
            .map(|layer| (layer.weight_ih_snapshot(), layer.weight_hh_snapshot()))
    }

    /// Borrow the controller network for read-only inspection.
    pub fn network_snapshot(&self) -> &LSTMNetwork<T> {
        &self.lstm_network
    }
}
